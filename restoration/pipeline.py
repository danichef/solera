from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage.segment import segment_faces
from damage.silver_wear import SilverWearFilter
from damage.clipping import ChipFilter

PROMPT = "restore the ancient silver coin to its original condition"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"

SAND_GRADES = ["f", "vg", "g", "ag"]

CHIP_NONE_PROB = 0.20
BIG_CLIP_PROB = 0.15
CHIP_RANGES = {
    "amplitude": (0.03, 0.09),
    "threshold": (0.40, 0.62),
    "waviness": (0.005, 0.013),
}

META_COLUMNS = [
    "coin_id", "face", "split", "sand_grade", "seed",
    "has_chips", "chip_amplitude", "chip_threshold", "chip_waviness",
    "big_clip", "n_big_chips", "n_faces",
    "clean_path", "mask_path", "damaged_path",
]

_WORKER_CFG = {}


# Draws chip parameters independently of the sanding grade, as chips on real
# coins do not correlate with wear.
# @params rng: numpy random generator
# @output dict of ChipFilter overrides, or None for a chip-free coin
def sample_chip_params(rng):
    if rng.random() < CHIP_NONE_PROB:
        return None

    params = {}
    for name, (low, high) in CHIP_RANGES.items():
        params[name] = float(rng.uniform(low, high))

    if rng.random() < BIG_CLIP_PROB:
        params["big_chip_prob"] = 1.0
    else:
        params["big_chip_prob"] = 0.0

    return params


# Applies grade-based sanding wear and then independently sampled rim chips
# to a single coin face.
# @params image: float RGB face crop in [0, 1]
# @params coin_mask: float mask marking coin pixels
# @params sand_grade: one of SAND_GRADES
# @params seed: seed for this damage variant
# @output (damaged image, metadata record)
def damage_face(image, coin_mask, sand_grade, seed):
    rng = np.random.default_rng(seed)
    worn = SilverWearFilter.for_grade(sand_grade).apply(image, coin_mask, seed=seed)
    chip_params = sample_chip_params(rng)

    record = {
        "sand_grade": sand_grade,
        "seed": seed,
        "has_chips": chip_params is not None,
        "chip_amplitude": np.nan,
        "chip_threshold": np.nan,
        "chip_waviness": np.nan,
        "big_clip": False,
        "n_big_chips": 0,
    }

    if chip_params is None:
        return worn.image, record

    chipped = ChipFilter(**chip_params).apply(worn.image, worn.coin_mask, seed=seed + 1)
    big_chips = chipped.params.get("big_chips") or []

    record["chip_amplitude"] = round(chip_params["amplitude"], 4)
    record["chip_threshold"] = round(chip_params["threshold"], 4)
    record["chip_waviness"] = round(chip_params["waviness"], 4)
    record["big_clip"] = chip_params["big_chip_prob"] > 0
    record["n_big_chips"] = len(big_chips)

    return chipped.image, record


# Assigns a coin to train/val/test by hashing its id, so adding more coins
# later never moves an existing coin between splits.
# @params coin_id: filename stem of the coin scan
# @output "train", "val" or "test"
def stable_split(coin_id):
    bucket = int(hashlib.md5(coin_id.encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


# Derives a reproducible seed from any mix of identifiers.
# @params parts: values that identify one damage variant
# @output 32-bit integer seed
def stable_seed(*parts):
    key = "_".join(str(part) for part in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# Pads an array to a square canvas and resizes it to the target resolution.
# @params img: float image or mask in [0, 1]
# @params res: output side length in pixels
# @params fill: padding value (white for images, black for masks)
# @params resample: PIL resampling filter
# @output uint8 array of shape (res, res[, 3])
def to_square(img, res, fill, resample):
    height, width = img.shape[:2]
    side = max(height, width)

    if img.ndim == 3:
        canvas = np.full((side, side, img.shape[2]), fill, dtype=np.float32)
    else:
        canvas = np.full((side, side), fill, dtype=np.float32)

    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top:top + height, left:left + width] = img

    as_uint8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(as_uint8).resize((res, res), resample))


# Saves a float image as a square JPEG on a white background.
# @params arr01: float RGB image in [0, 1]
# @params path: destination file
# @params res: output resolution
def save_jpg(arr01, path, res):
    squared = to_square(arr01, res, fill=1.0, resample=Image.LANCZOS)
    Image.fromarray(squared).save(path, quality=95)


# Saves a coin mask as a square binary PNG.
# @params mask: float mask in [0, 1]
# @params path: destination file
# @params res: output resolution
def save_mask(mask, path, res):
    squared = to_square(mask, res, fill=0.0, resample=Image.BILINEAR)
    binary = ((squared > 127) * 255).astype(np.uint8)
    Image.fromarray(binary).save(path)


# Receives the shared config in each worker process of the generation pool.
# @params cfg: dict with the output directory and resolution
def _init_worker(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg


# Processes one coin scan inside a worker: segments both faces, saves the
# clean target and mask, and renders all four damage grades per face.
# @params jpg_path_str: path to the source scan
# @output ("ok", coin_id, metadata rows) or ("error", coin_id, message)
def _process_coin(jpg_path_str):
    out = Path(_WORKER_CFG["out"])
    res = _WORKER_CFG["res"]
    jpg_path = Path(jpg_path_str)
    coin_id = jpg_path.stem
    split = stable_split(coin_id)

    try:
        rgb = np.array(Image.open(jpg_path).convert("RGB"))
        faces = segment_faces(rgb)
        if not faces:
            return "error", coin_id, "no faces found"

        if len(faces) > 2:
            faces = sorted(faces, key=lambda f: f.coin_mask.sum(), reverse=True)[:2]
            faces = sorted(faces, key=lambda f: f.bbox[0])

        rows = []
        for face_idx, face in enumerate(faces):
            clean_rel = f"{split}/clean/{coin_id}_{face_idx}.jpg"
            mask_rel = f"{split}/masks/{coin_id}_{face_idx}.png"
            save_jpg(face.image, out / clean_rel, res)
            save_mask(face.coin_mask, out / mask_rel, res)

            for grade in SAND_GRADES:
                seed = stable_seed(coin_id, face_idx, grade)
                damaged, record = damage_face(face.image, face.coin_mask, grade, seed)
                damaged_rel = f"{split}/damaged/{coin_id}_{face_idx}_{grade}.jpg"
                save_jpg(damaged, out / damaged_rel, res)

                row = dict(record)
                row.update(coin_id=coin_id, face=face_idx, split=split,
                           n_faces=len(faces), clean_path=clean_rel,
                           mask_path=mask_rel, damaged_path=damaged_rel)
                rows.append(row)

        return "ok", coin_id, rows

    except Exception as exc:
        return "error", coin_id, f"{type(exc).__name__}: {exc}"


# Renders a quick visual check: one row per face showing the clean image
# next to its four damage grades.
# @params out: dataset root containing the saved images
# @params rows: metadata rows to draw from
# @params n_faces: how many faces to include
def make_contact_sheet(out, rows, n_faces=6):
    groups = []
    seen = set()
    for row in rows:
        key = (row["coin_id"], row["face"])
        if key not in seen:
            seen.add(key)
            groups.append(key)
        if len(groups) >= n_faces:
            break

    cells = []
    for coin_id, face in groups:
        face_rows = [r for r in rows if (r["coin_id"], r["face"]) == (coin_id, face)]
        images = [Image.open(out / face_rows[0]["clean_path"])]
        for grade in SAND_GRADES:
            match = next(r for r in face_rows if r["sand_grade"] == grade)
            images.append(Image.open(out / match["damaged_path"]))
        cells.append(images)

    if not cells:
        return

    cell_size = cells[0][0].width
    sheet = Image.new("RGB", (cell_size * 5, cell_size * len(cells)), (255, 255, 255))
    for row_idx, images in enumerate(cells):
        for col_idx, image in enumerate(images):
            sheet.paste(image, (col_idx * cell_size, row_idx * cell_size))
    sheet.save(out / "contact_sheet.png")


class CoinRestorationPipeline:

    # Holds the parsed CLI config; every stage reads its settings from here.
    # @params cfg: argparse namespace produced by main()
    def __init__(self, cfg):
        self.cfg = cfg

    # Runs everything hands-off: a tiny smoke validation first, then the full
    # pipeline including an automatic guidance sweep before the test pass.
    # Every phase skips itself when its output already exists, so rerunning
    # this after any interruption just continues.
    # @output nothing; prints a summary with all artifact locations
    def run(self):
        started = time.time()

        if not self.cfg.skip_smoke:
            self._run_smoke_validation()

        self._run_stages()

        hours = (time.time() - started) / 3600
        print(f"\npipeline complete in {hours:.2f} h")
        print(f"  dataset: {self.cfg.data}")
        print(f"  model:   {Path(self.cfg.output_dir) / 'final'}")
        print(f"  results: {self.cfg.results}")
        print("next: download the results tar and run 'pipeline.py eval' on your machine")

    # Runs the whole pipeline in miniature (20 coins, 100 steps) into *_smoke
    # paths, so a config or data problem surfaces in minutes instead of hours.
    # @output nothing; raises if any stage fails
    def _run_smoke_validation(self):
        original = self.cfg
        smoke = argparse.Namespace(**vars(original))
        smoke.num_coins = 20
        smoke.max_steps = 100
        smoke.val_every = 50
        smoke.checkpoint_every = 100
        smoke.limit = 40
        smoke.no_sweep = True
        smoke.resume = None
        smoke.push_to_hub = None
        smoke.data = self._smoke_path(original.data)
        smoke.output_dir = self._smoke_path(original.output_dir)
        smoke.results = self._smoke_path(original.results)

        if (Path(smoke.results) / "results_metadata.csv").exists():
            print("smoke validation already passed, continuing to the full run")
            return

        print("SMOKE VALIDATION: tiny end-to-end run into *_smoke paths")
        self.cfg = smoke
        try:
            self._run_stages()
        finally:
            self.cfg = original
        print("smoke validation passed, starting the full run")

    # Executes the four stages with skip logic and training auto-resume.
    # @output nothing
    def _run_stages(self):
        cfg = self.cfg
        data = Path(cfg.data)
        final = Path(cfg.output_dir) / "final"
        results = Path(cfg.results)

        if (data / "metadata.csv").exists():
            self._banner(1, "damage generation", f"dataset found at {data}, skipping")
        else:
            self._banner(1, "damage generation")
            self._timed(self.prepare_data)

        if (final / "model_index.json").exists():
            self._banner(2, "training", f"trained model found at {final}, skipping")
        else:
            checkpoints = sorted(Path(cfg.output_dir).glob("checkpoint-*"),
                                 key=lambda d: int(d.name.split("-")[1]))
            if checkpoints:
                cfg.resume = str(checkpoints[-1])
                print(f"resuming from {checkpoints[-1].name}")
            self._banner(2, "training")
            self._timed(self.train)

        if cfg.no_sweep:
            self._banner(3, "guidance sweep", "disabled, using configured guidance")
        elif (results / "results_metadata.csv").exists():
            self._banner(3, "guidance sweep", "results already exist, skipping")
        else:
            self._banner(3, "guidance sweep")
            cfg.guidance = self._pick_guidance()

        if (results / "results_metadata.csv").exists():
            self._banner(4, "test inference", f"results found at {results}, skipping")
        else:
            self._banner(4, "test inference")
            self._timed(self.infer)

    # Restores a small val subset at several guidance scales and keeps the one
    # with the best masked PSNR, so the test pass never runs on a bad default.
    # The choice is cached in guidance.json and reused on reruns.
    # @output the winning guidance scale
    def _pick_guidance(self):
        cfg = self.cfg
        cache = Path(cfg.output_dir) / "guidance.json"
        if cache.exists():
            chosen = json.loads(cache.read_text())
            print(f"cached sweep result: guidance {chosen['guidance']}")
            return chosen["guidance"]

        candidates = (4.0, 6.0, 8.0)
        scores = {}
        for guidance in candidates:
            sweep_dir = Path(cfg.output_dir) / f"sweep_g{guidance}"
            if not (sweep_dir / "results_metadata.csv").exists():
                self.infer(override_split="val", override_limit=100,
                           override_guidance=guidance, override_out=sweep_dir,
                           override_tar=False)
            scores[guidance] = self._mean_masked_psnr(sweep_dir)
            print(f"guidance {guidance}: masked PSNR {scores[guidance]:.2f} dB")

        best = max(scores, key=scores.get)
        cache.write_text(json.dumps({"guidance": best, "scores": scores}))
        print(f"picked guidance {best}")
        return best

    # Scores one sweep folder: average masked PSNR of restored vs clean.
    # @params results_dir: folder written by infer()
    # @output mean PSNR in dB
    def _mean_masked_psnr(self, results_dir):
        data = Path(self.cfg.data)
        res = self.cfg.resolution
        with open(Path(results_dir) / "results_metadata.csv") as f:
            rows = list(csv.DictReader(f))

        values = []
        for row in rows:
            clean = self._load01(data / row["clean_path"], res)
            restored = self._load01(Path(results_dir) / row["restored_path"], res)
            mask_img = Image.open(data / row["mask_path"]).convert("L")
            mask = np.asarray(mask_img.resize((res, res), Image.BILINEAR),
                              dtype=np.float32) / 255.0
            values.append(self._masked_psnr(clean, restored, mask))
        return float(np.nanmean(values))

    # Generates the paired dataset: for every coin, both faces are segmented
    # and each face is damaged at all four sanding grades.
    # @output dataset folder with train/val/test splits and metadata.csv
    def prepare_data(self):
        cfg = self.cfg
        src = Path(cfg.src)
        out = Path(cfg.data)

        jpgs = sorted(src.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(jpgs)
        jpgs = jpgs[:cfg.num_coins]
        print(f"{len(jpgs)} coins from {src} -> {out}")

        for split in ("train", "val", "test"):
            for sub in ("clean", "masks", "damaged"):
                (out / split / sub).mkdir(parents=True, exist_ok=True)

        worker_cfg = {"out": str(out), "res": cfg.resolution}
        all_rows = []
        errors = []

        with Pool(cfg.workers, initializer=_init_worker, initargs=(worker_cfg,)) as pool, \
                tqdm(total=len(jpgs), desc="damaging coins", unit="coin") as bar:
            jobs = [str(path) for path in jpgs]
            for status, coin_id, payload in pool.imap_unordered(_process_coin, jobs,
                                                                chunksize=4):
                if status == "ok":
                    all_rows.extend(payload)
                else:
                    errors.append((coin_id, payload))
                bar.update(1)
                bar.set_postfix(pairs=len(all_rows), errors=len(errors))

        all_rows.sort(key=lambda r: (r["coin_id"], r["face"], r["sand_grade"]))
        with open(out / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=META_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        split_counts = {}
        for split in ("train", "val", "test"):
            split_counts[split] = len({r["coin_id"] for r in all_rows
                                       if r["split"] == split})
        print(f"rows={len(all_rows)}  coins per split={split_counts}  errors={len(errors)}")
        for coin_id, message in errors[:10]:
            print(f"  ERROR {coin_id}: {message}")

        make_contact_sheet(out, all_rows)
        print(f"-> {out / 'contact_sheet.png'}")

        if cfg.tar:
            tar_path = out / "dataset.tar"
            with tarfile.open(tar_path, "w") as tar:
                for split in ("train", "val", "test"):
                    tar.add(out / split, arcname=split)
                tar.add(out / "metadata.csv", arcname="metadata.csv")
            print(f"-> {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")

    # Fine-tunes SDXL as an InstructPix2Pix restorer with a masked latent loss
    # and EMA weights; saves the final pipeline and optionally pushes it to HF.
    # @output fine-tuned model at <output-dir>/final
    def train(self):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import EMAModel

        cfg = self.cfg
        accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum,
                                  mixed_precision=cfg.mixed_precision)
        set_seed(cfg.seed)

        out_dir = Path(cfg.output_dir)
        if accelerator.is_main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "previews").mkdir(exist_ok=True)
            (out_dir / "train_args.json").write_text(json.dumps(vars(cfg), indent=2))

        weight_dtype = {"no": torch.float32,
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16}[cfg.mixed_precision]
        device = accelerator.device

        components = self._load_sdxl_components()
        unet = components["unet"]
        vae = components["vae"]
        noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

        self._expand_conv_in(unet)
        vae.requires_grad_(False)
        components["text_encoder"].requires_grad_(False)
        components["text_encoder_2"].requires_grad_(False)
        if cfg.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        torch.backends.cuda.matmul.allow_tf32 = True

        vae.to(device, dtype=torch.float32)
        components["text_encoder"].to(device, dtype=weight_dtype)
        components["text_encoder_2"].to(device, dtype=weight_dtype)

        train_dataset = self._dataset("train")
        val_rows = self._metadata_rows("val")[:4]
        loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.workers, pin_memory=True, drop_last=True,
                            persistent_workers=cfg.workers > 0)
        accelerator.print(f"train pairs: {len(train_dataset)}")

        optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.lr,
                                      betas=(0.9, 0.999), weight_decay=1e-2)
        lr_schedule = get_scheduler("cosine", optimizer,
                                    num_warmup_steps=cfg.warmup_steps,
                                    num_training_steps=cfg.max_steps)

        unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
        accelerator.register_for_checkpointing(lr_schedule)
        components["unet"] = unet

        ema = EMAModel(accelerator.unwrap_model(unet).parameters(), decay=0.9999,
                       model_cls=type(accelerator.unwrap_model(unet)),
                       model_config=accelerator.unwrap_model(unet).config)
        ema.to(device)

        prompt_emb, prompt_pooled = self._encode_prompt([PROMPT], components, device)
        null_emb, null_pooled = self._encode_prompt([""], components, device)
        res = cfg.resolution
        base_time_ids = torch.tensor([res, res, 0, 0, res, res], device=device)

        global_step = 0
        if cfg.resume:
            accelerator.load_state(cfg.resume)
            ema_path = Path(cfg.resume) / "ema.pt"
            if ema_path.exists():
                ema.load_state_dict(torch.load(ema_path, map_location=device))
            global_step = int(Path(cfg.resume).name.split("-")[1])

        scaling = vae.config.scaling_factor
        dropout = cfg.conditioning_dropout
        progress = tqdm(total=cfg.max_steps, initial=global_step,
                        desc="training", unit="step",
                        disable=not accelerator.is_main_process)

        log_path = out_dir / "train_log.csv"
        if accelerator.is_main_process and not log_path.exists():
            log_path.write_text("step,loss,lr\n")

        unet.train()
        done = False
        while not done:
            for batch in loader:
                with accelerator.accumulate(unet):
                    with torch.no_grad():
                        latents = vae.encode(batch["target"].float()).latent_dist.sample()
                        latents = latents * scaling
                        cond_latents = vae.encode(batch["input"].float()).latent_dist.mode()

                    noise = torch.randn_like(latents)
                    batch_size = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (batch_size,), device=device).long()
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                    emb = prompt_emb.repeat(batch_size, 1, 1)
                    pooled = prompt_pooled.repeat(batch_size, 1)
                    if dropout > 0:
                        draw = torch.rand(batch_size, device=device)
                        drop_text = draw < 2 * dropout
                        emb = torch.where(drop_text[:, None, None],
                                          null_emb.repeat(batch_size, 1, 1), emb)
                        pooled = torch.where(drop_text[:, None],
                                             null_pooled.repeat(batch_size, 1), pooled)
                        drop_image = (draw >= dropout) & (draw < 3 * dropout)
                        drop_image = drop_image.float()[:, None, None, None]
                        cond_latents = cond_latents * (1.0 - drop_image)

                    model_input = torch.cat([noisy, cond_latents], dim=1)
                    added_cond = {
                        "text_embeds": pooled.to(weight_dtype),
                        "time_ids": base_time_ids.repeat(batch_size, 1).to(weight_dtype),
                    }
                    prediction = unet(model_input.to(weight_dtype), timesteps,
                                      encoder_hidden_states=emb.to(weight_dtype),
                                      added_cond_kwargs=added_cond).sample

                    mask = F.interpolate(batch["mask"], size=prediction.shape[-2:],
                                         mode="area")
                    weights = cfg.mask_floor + (1.0 - cfg.mask_floor) * mask
                    squared_error = (prediction.float() - noise.float()) ** 2
                    loss = (squared_error * weights.float()).sum()
                    loss = loss / (weights.float().sum() * squared_error.shape[1] + 1e-8)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    ema.step(accelerator.unwrap_model(unet).parameters())
                    lr_schedule.step()
                    global_step += 1
                    progress.update(1)
                    is_main = accelerator.is_main_process
                    if global_step % 10 == 0:
                        current_loss = loss.item()
                        current_lr = lr_schedule.get_last_lr()[0]
                        progress.set_postfix(loss=f"{current_loss:.4f}",
                                             lr=f"{current_lr:.1e}")
                        if is_main:
                            with open(log_path, "a") as log:
                                log.write(f"{global_step},{current_loss:.6f},"
                                          f"{current_lr:.6e}\n")

                    if is_main and global_step % cfg.val_every == 0:
                        self._save_preview(components, accelerator, ema, val_rows,
                                           out_dir / "previews" / f"step_{global_step}.png",
                                           weight_dtype, device)
                        tqdm.write(f"preview saved at step {global_step}")
                    if is_main and global_step % cfg.checkpoint_every == 0:
                        self._save_checkpoint(accelerator, ema, out_dir, global_step)
                        tqdm.write(f"checkpoint saved at step {global_step}")
                    if global_step >= cfg.max_steps:
                        done = True
                        break

        progress.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            ema.copy_to(accelerator.unwrap_model(unet).parameters())
            pipe = self._build_sdxl_pipe(components, accelerator)
            if cfg.mixed_precision != "no":
                pipe.to(torch.float16)
            final_dir = out_dir / "final"
            pipe.save_pretrained(final_dir)
            print(f"model saved -> {final_dir}")
            if cfg.push_to_hub:
                self._push_to_hub(final_dir)

    # Restores every image of one split with the trained model and writes
    # triptychs plus a metadata csv recording what state each came from.
    # The override arguments let the guidance sweep reuse this with different
    # settings without touching the main config.
    # @params override_split / override_limit / override_guidance /
    #         override_out / override_tar: optional one-off settings
    # @output results folder with restored/, triptychs/, results_metadata.csv
    def infer(self, override_split=None, override_limit=None,
              override_guidance=None, override_out=None, override_tar=None):
        import torch
        from diffusers import DiffusionPipeline

        cfg = self.cfg
        data = Path(cfg.data)
        out = Path(override_out) if override_out else Path(cfg.results)
        split = override_split or cfg.split
        limit = override_limit if override_limit is not None else cfg.limit
        guidance = override_guidance if override_guidance is not None else cfg.guidance
        make_tar = override_tar if override_tar is not None else cfg.results_tar
        (out / "restored").mkdir(parents=True, exist_ok=True)
        (out / "triptychs").mkdir(exist_ok=True)

        model_path = cfg.model or str(Path(cfg.output_dir) / "final")
        rows = self._metadata_rows(split)
        if limit:
            rows = rows[:limit]
        print(f"{len(rows)} {split} samples, model: {model_path}, guidance {guidance}")

        pipe = DiffusionPipeline.from_pretrained(model_path,
                                                 torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)

        results = []
        bar = tqdm(range(0, len(rows), cfg.batch), desc="restoring", unit="batch")
        for start in bar:
            chunk = rows[start:start + cfg.batch]
            damaged = [Image.open(data / r["damaged_path"]).convert("RGB")
                       for r in chunk]
            generators = [torch.Generator("cuda").manual_seed(int(r["seed"]))
                          for r in chunk]

            restored_batch = pipe([PROMPT] * len(chunk), image=damaged,
                                  height=cfg.resolution, width=cfg.resolution,
                                  num_inference_steps=cfg.steps,
                                  image_guidance_scale=cfg.image_guidance,
                                  guidance_scale=guidance,
                                  generator=generators).images

            for row, damaged_img, restored in zip(chunk, damaged, restored_batch):
                stem = f"{row['coin_id']}_{row['face']}_{row['sand_grade']}"
                restored_rel = f"restored/{stem}.png"
                restored.save(out / restored_rel)

                clean = Image.open(data / row["clean_path"]).convert("RGB")
                width, height = restored.size
                triptych = Image.new("RGB", (width * 3, height), (255, 255, 255))
                triptych.paste(damaged_img.resize((width, height)), (0, 0))
                triptych.paste(restored, (width, 0))
                triptych.paste(clean.resize((width, height)), (width * 2, 0))
                triptych_rel = f"triptychs/{stem}.jpg"
                triptych.save(out / triptych_rel, quality=92)

                results.append(dict(row, restored_path=restored_rel,
                                    triptych_path=triptych_rel, steps=cfg.steps,
                                    image_guidance=cfg.image_guidance,
                                    guidance=guidance, model=model_path))
            bar.set_postfix(images=len(results))

        with open(out / "results_metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        if make_tar:
            tar_path = out.parent / f"{out.name}.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(out, arcname=out.name)
            print(f"-> {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")

    # Scores the downloaded results against the clean targets with masked
    # PSNR / SSIM / LPIPS, grouped by the damage state each image came from.
    # @output results_per_image.csv and results_summary.csv next to the results
    def evaluate(self):
        import pandas as pd
        from skimage.metrics import structural_similarity

        cfg = self.cfg
        data = Path(cfg.data)
        results = Path(cfg.results)
        res = cfg.resolution

        df = pd.read_csv(results / "results_metadata.csv")
        df["has_chips"] = df["has_chips"].astype(str).str.lower() == "true"
        df["big_clip"] = df["big_clip"].astype(str).str.lower() == "true"
        df["chip_bin"] = df.apply(self._chip_bin, axis=1)
        print(f"{len(df)} results")

        lpips_scorer = None
        if not cfg.no_lpips:
            try:
                lpips_scorer = self._lpips_scorer(cfg.device)
            except ImportError:
                print("lpips not installed, skipping (pip install lpips torch)")

        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="scoring", unit="img"):
            clean = self._load01(data / row["clean_path"], res)
            restored = self._load01(results / row["restored_path"], res)
            damaged = self._load01(data / row["damaged_path"], res)
            mask_img = Image.open(data / row["mask_path"]).convert("L")
            mask = np.asarray(mask_img.resize((res, res), Image.BILINEAR),
                              dtype=np.float32) / 255.0

            record = {
                "psnr": self._masked_psnr(clean, restored, mask),
                "ssim": self._masked_ssim(clean, restored, mask, structural_similarity),
                "psnr_unmasked": self._masked_psnr(clean, restored, np.ones_like(mask)),
                "psnr_damaged": self._masked_psnr(clean, damaged, mask),
            }
            if lpips_scorer:
                record["lpips"] = lpips_scorer(clean, restored, mask)
                record["lpips_damaged"] = lpips_scorer(clean, damaged, mask)
            records.append(record)

        metrics = pd.DataFrame(records)
        full = pd.concat([df, metrics], axis=1)
        full.to_csv(results / "results_per_image.csv", index=False)

        columns = [c for c in ("psnr", "ssim", "lpips", "psnr_damaged", "lpips_damaged")
                   if c in full.columns]
        by_grade = full.groupby("sand_grade")[columns].agg(["mean", "std"]).round(4)
        by_grade = by_grade.reindex(SAND_GRADES)
        by_chip = full.groupby("chip_bin")[columns].agg(["mean", "std"]).round(4)
        by_chip = by_chip.reindex(["none", "light", "medium", "heavy"])
        by_clip = full.groupby("big_clip")[columns].agg(["mean", "std"]).round(4)

        print("\n== by sanding grade ==\n", by_grade)
        print("\n== by chip severity ==\n", by_chip)
        print("\n== by big flan clip ==\n", by_clip)

        with open(results / "results_summary.csv", "w") as f:
            f.write("# by sanding grade\n")
            by_grade.to_csv(f)
            f.write("\n# by chip severity\n")
            by_chip.to_csv(f)
            f.write("\n# by big flan clip\n")
            by_clip.to_csv(f)

        fid = None if cfg.no_fid else self._compute_fid()
        self._make_plots(full, results)
        self._write_report(full, by_grade, by_chip, by_clip, fid, results)

        print(f"\n-> {results / 'results_per_image.csv'}")
        print(f"-> {results / 'results_summary.csv'}")
        print(f"-> {results / 'results_report.md'}")
        print(f"-> {results / 'plots/'}")

    # Computes distribution-level FID: restored vs clean, plus the damaged vs
    # clean baseline that shows how far restoration moved the distribution.
    # @output dict with both scores, or None when pytorch-fid is missing
    def _compute_fid(self):
        try:
            from pytorch_fid import fid_score
        except ImportError:
            print("pytorch-fid not installed, skipping FID (pip install pytorch-fid)")
            return None

        cfg = self.cfg
        clean_dir = str(Path(cfg.data) / "test" / "clean")
        damaged_dir = str(Path(cfg.data) / "test" / "damaged")
        restored_dir = str(Path(cfg.results) / "restored")

        print("computing FID (restored vs clean)")
        fid_restored = fid_score.calculate_fid_given_paths(
            [restored_dir, clean_dir], batch_size=32, device=cfg.device, dims=2048)
        print("computing FID baseline (damaged vs clean)")
        fid_damaged = fid_score.calculate_fid_given_paths(
            [damaged_dir, clean_dir], batch_size=32, device=cfg.device, dims=2048)

        print(f"FID restored: {fid_restored:.2f}   FID damaged: {fid_damaged:.2f}")
        return {"restored": fid_restored, "damaged": fid_damaged}

    # Draws the evaluation figures: metrics by damage state, the per-image
    # improvement scatter, metric distributions, and the training loss curve.
    # @params full: per-image dataframe with metric columns
    # @params results: results folder; figures land in results/plots
    def _make_plots(self, full, results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = Path(results) / "plots"
        plots.mkdir(exist_ok=True)

        restored_color = "#2a78d6"
        damaged_color = "#8a8983"
        grade_colors = {"f": "#2a78d6", "vg": "#1baf7a",
                        "g": "#eda100", "ag": "#4a3aa7"}
        text_color = "#0b0b0b"
        plt.rcParams.update({"font.size": 11, "text.color": text_color,
                             "axes.labelcolor": text_color,
                             "xtick.color": "#52514e", "ytick.color": "#52514e",
                             "axes.edgecolor": "#d5d4cf", "figure.dpi": 150})

        def style(ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
            ax.set_axisbelow(True)

        def grouped_bars(path, groups, order, title_suffix):
            metrics = [("psnr", "psnr_damaged", "PSNR (dB, higher better)"),
                       ("ssim", None, "SSIM (higher better)"),
                       ("lpips", "lpips_damaged", "LPIPS (lower better)")]
            metrics = [(m, b, t) for m, b, t in metrics if m in full.columns]
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
            for ax, (metric, baseline, title) in zip(np.atleast_1d(axes), metrics):
                means = groups[metric].mean().reindex(order)
                stds = groups[metric].std().reindex(order)
                x = np.arange(len(order))
                width = 0.38 if baseline else 0.6
                ax.bar(x, means, width, yerr=stds, capsize=3,
                       color=restored_color, label="restored")
                if baseline and baseline in full.columns:
                    base_means = groups[baseline].mean().reindex(order)
                    base_stds = groups[baseline].std().reindex(order)
                    ax.bar(x + width, base_means, width, yerr=base_stds, capsize=3,
                           color=damaged_color, label="damaged")
                    ax.set_xticks(x + width / 2)
                else:
                    ax.set_xticks(x)
                ax.set_xticklabels(order)
                ax.set_title(f"{title}\nby {title_suffix}", fontsize=11)
                style(ax)
            first = np.atleast_1d(axes)[0]
            if first.get_legend_handles_labels()[0]:
                first.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        grouped_bars(plots / "metrics_by_grade.png",
                     full.groupby("sand_grade"), SAND_GRADES, "sanding grade")
        grouped_bars(plots / "metrics_by_chip.png",
                     full.groupby("chip_bin"),
                     ["none", "light", "medium", "heavy"], "chip severity")

        if "lpips" in full.columns and "lpips_damaged" in full.columns:
            fig, ax = plt.subplots(figsize=(5.4, 5.4))
            for grade in SAND_GRADES:
                sub = full[full["sand_grade"] == grade]
                ax.scatter(sub["lpips_damaged"], sub["lpips"], s=8, alpha=0.5,
                           color=grade_colors[grade], label=grade, linewidths=0)
            limit = float(max(full["lpips_damaged"].max(), full["lpips"].max())) * 1.05
            ax.plot([0, limit], [0, limit], color="#b4b2a9", linewidth=1,
                    linestyle="--")
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_xlabel("LPIPS damaged vs clean")
            ax.set_ylabel("LPIPS restored vs clean")
            ax.set_title("Per-image improvement (below the line = improved)",
                         fontsize=11)
            ax.legend(title="grade", frameon=False)
            style(ax)
            ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
            fig.tight_layout()
            fig.savefig(plots / "lpips_improvement.png", bbox_inches="tight")
            plt.close(fig)

        pairs = [("psnr", "psnr_damaged", "PSNR (dB)"),
                 ("lpips", "lpips_damaged", "LPIPS")]
        pairs = [(m, b, t) for m, b, t in pairs
                 if m in full.columns and b in full.columns]
        if pairs:
            fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 3.6))
            for ax, (metric, baseline, title) in zip(np.atleast_1d(axes), pairs):
                ax.hist(full[baseline].dropna(), bins=40, color=damaged_color,
                        alpha=0.6, label="damaged")
                ax.hist(full[metric].dropna(), bins=40, color=restored_color,
                        alpha=0.6, label="restored")
                ax.set_xlabel(title)
                ax.set_ylabel("images")
                ax.legend(frameon=False)
                style(ax)
            fig.tight_layout()
            fig.savefig(plots / "distributions.png", bbox_inches="tight")
            plt.close(fig)

        if self.cfg.train_log and Path(self.cfg.train_log).exists():
            import pandas as pd
            log = pd.read_csv(self.cfg.train_log)
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            ax.plot(log["step"], log["loss"], color="#9ec5f4", linewidth=0.7)
            smooth = log["loss"].rolling(100, min_periods=1).mean()
            ax.plot(log["step"], smooth, color="#1c5cab", linewidth=1.8,
                    label="rolling mean (100)")
            ax.set_xlabel("step")
            ax.set_ylabel("training loss")
            ax.legend(frameon=False)
            style(ax)
            fig.tight_layout()
            fig.savefig(plots / "training_loss.png", bbox_inches="tight")
            plt.close(fig)

    # Writes a human-readable evaluation report next to the csv outputs.
    # @params full: per-image dataframe
    # @params by_grade / by_chip / by_clip: grouped summary tables
    # @params fid: FID scores dict or None
    # @params results: results folder
    def _write_report(self, full, by_grade, by_chip, by_clip, fid, results):
        def stat(column):
            if column not in full.columns:
                return "n/a"
            return f"{full[column].mean():.4f} ± {full[column].std():.4f}"

        lines = ["# Solera evaluation report", ""]
        lines.append(f"Test images scored: {len(full)} "
                     f"({full['coin_id'].nunique()} coins)")
        lines.append("All PSNR / SSIM / LPIPS values are masked to the original "
                     "coin silhouette. 'Damaged' columns score the unrestored "
                     "input against the same target, as the baseline.")
        lines.append("")
        lines.append("## Overall")
        lines.append("")
        lines.append("| metric | restored | damaged baseline |")
        lines.append("|---|---|---|")
        lines.append(f"| PSNR (dB, higher better) | {stat('psnr')} "
                     f"| {stat('psnr_damaged')} |")
        lines.append(f"| SSIM (higher better) | {stat('ssim')} | n/a |")
        lines.append(f"| LPIPS (lower better) | {stat('lpips')} "
                     f"| {stat('lpips_damaged')} |")
        if fid:
            lines.append(f"| FID (lower better) | {fid['restored']:.2f} "
                         f"| {fid['damaged']:.2f} |")
        lines.append("")

        for name, table in (("by sanding grade", by_grade),
                            ("by chip severity", by_chip),
                            ("by big flan clip", by_clip)):
            lines.append(f"## Metrics {name}")
            lines.append("")
            lines.append(table.to_markdown())
            lines.append("")

        lines.append("## Files")
        lines.append("")
        lines.append("- `results_per_image.csv` — every metric for every image")
        lines.append("- `results_summary.csv` — the grouped tables above")
        lines.append("- `plots/metrics_by_grade.png`, `plots/metrics_by_chip.png` "
                     "— restored vs damaged by damage state")
        lines.append("- `plots/lpips_improvement.png` — per-image before/after")
        lines.append("- `plots/distributions.png` — metric histograms")
        if self.cfg.train_log and Path(self.cfg.train_log).exists():
            lines.append("- `plots/training_loss.png` — smoothed loss curve")

        (Path(results) / "results_report.md").write_text("\n".join(lines) + "\n")

    # Restores arbitrary coin images with a saved model; the entry point for
    # using the model after the project.
    # @output restored PNGs in cfg.results
    def restore(self):
        import torch
        from diffusers import DiffusionPipeline

        cfg = self.cfg
        device, dtype = self._pick_device()
        print(f"device={device} dtype={dtype}")

        pipe = DiffusionPipeline.from_pretrained(cfg.model, torch_dtype=dtype).to(device)

        source = Path(cfg.input)
        if source.is_dir():
            paths = sorted(p for p in source.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        else:
            paths = [source]

        out = Path(cfg.results)
        out.mkdir(parents=True, exist_ok=True)

        for path in paths:
            for suffix, img in self._input_faces(path):
                generator = torch.Generator(device).manual_seed(cfg.seed)
                restored = pipe(PROMPT, image=img,
                                height=cfg.resolution, width=cfg.resolution,
                                num_inference_steps=cfg.steps,
                                image_guidance_scale=cfg.image_guidance,
                                guidance_scale=cfg.guidance,
                                generator=generator).images[0]
                destination = out / f"{path.stem}{suffix}_restored.png"
                restored.save(destination)
                print(f"-> {destination}")

    # Prints a stage banner so the pipeline output is easy to scan.
    # @params idx: stage number
    # @params name: stage name
    # @params note: optional skip reason
    def _banner(self, idx, name, note=""):
        line = "=" * 64
        suffix = f"  —  {note}" if note else ""
        print(f"\n{line}\n  [{idx}/4] {name}{suffix}\n{line}", flush=True)

    # Runs one stage and reports how long it took.
    # @params stage: bound method to execute
    def _timed(self, stage):
        started = time.time()
        stage()
        minutes = (time.time() - started) / 60
        print(f"stage done in {minutes:.1f} min", flush=True)

    # Appends _smoke to a path so smoke runs never collide with real ones.
    # @params path_str: original path
    # @output adjusted path string
    def _smoke_path(self, path_str):
        path = Path(path_str)
        return str(path.with_name(path.name + "_smoke"))

    # Reads metadata rows for one split.
    # @params split: train / val / test
    # @output list of csv row dicts
    def _metadata_rows(self, split):
        with open(Path(self.cfg.data) / "metadata.csv") as f:
            return [row for row in csv.DictReader(f) if row["split"] == split]

    # Builds the torch Dataset of (damaged, clean, mask) tensors for training.
    # @params split: train / val / test
    # @output torch Dataset
    def _dataset(self, split):
        import torch
        from torch.utils.data import Dataset

        root = Path(self.cfg.data)
        resolution = self.cfg.resolution
        rows = self._metadata_rows(split)

        def load_image(rel):
            image = Image.open(root / rel).convert("RGB")
            if image.size != (resolution, resolution):
                image = image.resize((resolution, resolution), Image.LANCZOS)
            arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        class Pairs(Dataset):
            def __len__(self):
                return len(rows)

            def __getitem__(self, index):
                row = rows[index]
                mask = Image.open(root / row["mask_path"]).convert("L")
                if mask.size != (resolution, resolution):
                    mask = mask.resize((resolution, resolution), Image.BILINEAR)
                mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
                return {
                    "input": load_image(row["damaged_path"]),
                    "target": load_image(row["clean_path"]),
                    "mask": torch.from_numpy(mask_arr)[None],
                }

        return Pairs()

    # Loads the SDXL tokenizers, text encoders, vae and unet.
    # @output dict of named components
    def _load_sdxl_components(self):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                                  CLIPTokenizer)

        return {
            "tokenizer": CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer"),
            "tokenizer_2": CLIPTokenizer.from_pretrained(BASE_MODEL,
                                                         subfolder="tokenizer_2"),
            "text_encoder": CLIPTextModel.from_pretrained(BASE_MODEL,
                                                          subfolder="text_encoder"),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
                BASE_MODEL, subfolder="text_encoder_2"),
            "vae": AutoencoderKL.from_pretrained(VAE_MODEL),
            "unet": UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet"),
        }

    # Widens the unet input from 4 to 8 channels so it can take the damaged
    # image latents next to the noisy latents; new weights start at zero.
    # @params unet: SDXL UNet2DConditionModel
    def _expand_conv_in(self, unet):
        import torch
        import torch.nn as nn

        if unet.config.in_channels == 8:
            return
        old = unet.conv_in
        new = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :4] = old.weight
            new.bias.copy_(old.bias)
        unet.conv_in = new
        unet.register_to_config(in_channels=8)

    # Encodes text with both SDXL encoders.
    # @params text: list with one prompt string
    # @params components: dict from _load_sdxl_components
    # @params device: target device
    # @output (sequence embeddings, pooled embeddings)
    def _encode_prompt(self, text, components, device):
        import torch

        tokenizers = (components["tokenizer"], components["tokenizer_2"])
        encoders = (components["text_encoder"], components["text_encoder_2"])

        embeds = []
        pooled = None
        with torch.no_grad():
            for tokenizer, encoder in zip(tokenizers, encoders):
                ids = tokenizer(text, max_length=tokenizer.model_max_length,
                                padding="max_length", truncation=True,
                                return_tensors="pt").input_ids.to(device)
                output = encoder(ids, output_hidden_states=True)
                if hasattr(output, "text_embeds"):
                    pooled = output.text_embeds
                embeds.append(output.hidden_states[-2])

        return torch.cat(embeds, dim=-1), pooled

    # Assembles an SDXL InstructPix2Pix pipeline around the current unet.
    # @params components: dict from _load_sdxl_components
    # @params accelerator: the active Accelerator
    # @output diffusers pipeline on the training device
    def _build_sdxl_pipe(self, components, accelerator):
        from diffusers import StableDiffusionXLInstructPix2PixPipeline

        return StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            BASE_MODEL,
            unet=accelerator.unwrap_model(components["unet"]),
            vae=components["vae"],
            text_encoder=components["text_encoder"],
            text_encoder_2=components["text_encoder_2"],
            tokenizer=components["tokenizer"],
            tokenizer_2=components["tokenizer_2"]).to(accelerator.device)

    # Renders a damaged | restored | clean grid on the EMA weights.
    # @params components / accelerator / ema: live training objects
    # @params val_rows: metadata rows to preview
    # @params out_path: destination png
    # @params autocast_dtype / device: precision context
    def _save_preview(self, components, accelerator, ema, val_rows, out_path,
                      autocast_dtype, device):
        import torch

        unet = accelerator.unwrap_model(components["unet"])
        unet.eval()
        ema.store(unet.parameters())
        ema.copy_to(unet.parameters())
        pipe = self._build_sdxl_pipe(components, accelerator)

        res = self.cfg.resolution
        data_root = Path(self.cfg.data)
        cells = []
        with torch.no_grad():
            for row in val_rows:
                damaged = Image.open(data_root / row["damaged_path"]).convert("RGB")
                damaged = damaged.resize((res, res))
                clean = Image.open(data_root / row["clean_path"]).convert("RGB")
                clean = clean.resize((res, res))
                with torch.autocast(device.type, dtype=autocast_dtype,
                                    enabled=autocast_dtype != torch.float32):
                    restored = pipe(PROMPT, image=damaged,
                                    height=res, width=res,
                                    num_inference_steps=20,
                                    image_guidance_scale=1.5, guidance_scale=6.0,
                                    generator=torch.Generator(device).manual_seed(0)
                                    ).images[0]
                cells.append((damaged, restored, clean))

        grid = Image.new("RGB", (res * 3, res * len(cells)), (255, 255, 255))
        for i, (a, b, c) in enumerate(cells):
            grid.paste(a, (0, i * res))
            grid.paste(b, (res, i * res))
            grid.paste(c, (res * 2, i * res))
        grid.save(out_path)

        del pipe
        ema.restore(unet.parameters())
        torch.cuda.empty_cache()
        unet.train()

    # Saves a rotating training checkpoint including the EMA weights.
    # @params accelerator / ema: live training objects
    # @params out_dir: training output directory
    # @params global_step: current step, used in the checkpoint name
    def _save_checkpoint(self, accelerator, ema, out_dir, global_step):
        import torch

        checkpoints = sorted(out_dir.glob("checkpoint-*"),
                             key=lambda d: int(d.name.split("-")[1]))
        while len(checkpoints) >= self.cfg.keep_checkpoints:
            shutil.rmtree(checkpoints.pop(0))
        checkpoint_dir = out_dir / f"checkpoint-{global_step}"
        accelerator.save_state(checkpoint_dir)
        torch.save(ema.state_dict(), checkpoint_dir / "ema.pt")

    # Uploads the final model to a private Hugging Face repo.
    # @params final_dir: folder with the saved pipeline
    def _push_to_hub(self, final_dir):
        from huggingface_hub import HfApi

        token = None
        for env_path in (Path(__file__).resolve().parents[1] / "keys.env",
                         Path("keys.env")):
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
                        token = line.split("=", 1)[1].strip()

        api = HfApi(token=token)
        api.create_repo(self.cfg.push_to_hub, private=True, exist_ok=True,
                        repo_type="model")
        api.upload_folder(repo_id=self.cfg.push_to_hub, folder_path=str(final_dir))
        print(f"pushed -> https://huggingface.co/{self.cfg.push_to_hub}")

    # Picks the best available inference device.
    # @output (device name, torch dtype)
    def _pick_device(self):
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float32
        return "cpu", torch.float32

    # Prepares one input file for restore(): either the whole image squared,
    # or each segmented face when --segment is set.
    # @params path: input image path
    # @output list of (name suffix, PIL image) pairs
    def _input_faces(self, path):
        res = self.cfg.resolution
        image = Image.open(path).convert("RGB")

        def squared(img):
            side = max(img.size)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            return canvas.resize((res, res), Image.LANCZOS)

        if not self.cfg.segment:
            return [("", squared(image))]

        faces = segment_faces(np.array(image))
        prepared = []
        for i, face in enumerate(faces):
            arr = (np.clip(face.image, 0, 1) * 255).astype(np.uint8)
            prepared.append((f"_face{i}", squared(Image.fromarray(arr))))
        return prepared

    # Bins the drawn chip amplitude into a readable severity label.
    # @params row: one results row
    # @output "none", "light", "medium" or "heavy"
    def _chip_bin(self, row):
        if not row["has_chips"]:
            return "none"
        amplitude = row["chip_amplitude"]
        if amplitude < 0.05:
            return "light"
        if amplitude < 0.07:
            return "medium"
        return "heavy"

    # Loads an image as float RGB in [0, 1] at the evaluation resolution.
    # @params path: image file
    # @params res: target resolution
    # @output float array (res, res, 3)
    def _load01(self, path, res):
        image = Image.open(path).convert("RGB")
        if image.size != (res, res):
            image = image.resize((res, res), Image.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

    # PSNR computed only over the coin mask.
    # @params a, b: float images in [0, 1]
    # @params mask: float mask
    # @output PSNR in dB
    def _masked_psnr(self, a, b, mask):
        denominator = mask.sum() * 3.0
        if denominator < 1:
            return float("nan")
        mse = float((((a - b) ** 2) * mask[..., None]).sum() / denominator)
        return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))

    # Mean of the SSIM map over the coin mask.
    # @params a, b: float images in [0, 1]
    # @params mask: float mask
    # @params ssim_fn: skimage structural_similarity
    # @output masked SSIM
    def _masked_ssim(self, a, b, mask, ssim_fn):
        _, ssim_map = ssim_fn(a, b, channel_axis=2, data_range=1.0, full=True)
        selected = mask > 0.5
        if not selected.any():
            return float("nan")
        return float(ssim_map[selected].mean())

    # Builds an LPIPS scorer that compares images composited over white
    # inside the coin mask.
    # @params device: torch device string
    # @output callable (clean, restored, mask) -> lpips distance
    def _lpips_scorer(self, device):
        import lpips
        import torch

        net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

        def score(a, b, mask):
            masked_a = a * mask[..., None] + (1.0 - mask[..., None])
            masked_b = b * mask[..., None] + (1.0 - mask[..., None])
            ta = torch.from_numpy(masked_a * 2 - 1).permute(2, 0, 1)[None]
            tb = torch.from_numpy(masked_b * 2 - 1).permute(2, 0, 1)[None]
            with torch.no_grad():
                return float(net(ta.float().to(device), tb.float().to(device)).item())

        return score


# Builds the CLI: one subcommand per stage plus "run" for the full pipeline.
# @output parsed argparse namespace
def parse_args():
    parser = argparse.ArgumentParser(
        description="coin restoration pipeline: damage -> train -> test")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p):
        p.add_argument("--data", default="data/restore_v1")
        p.add_argument("--resolution", type=int, default=768)
        p.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="the full automatic pipeline: smoke check, "
                                     "damage, training, guidance sweep, test")
    prepare = sub.add_parser("prepare", help="damage generation only")
    train = sub.add_parser("train", help="training only")
    infer = sub.add_parser("infer", help="inference over a split")
    evaluate = sub.add_parser("eval", help="score downloaded results locally")
    restore = sub.add_parser("restore", help="restore arbitrary images")

    for p in (run, prepare):
        add_shared(p)
        p.add_argument("--src", default=str(Path.home() / "Downloads/data_part_1"))
        p.add_argument("--num-coins", type=int, default=9999)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--tar", action="store_true",
                       help="also pack the dataset into dataset.tar for upload")

    for p in (run, train):
        p.add_argument("--output-dir", default="/workspace/coin-restorer-xl")
        p.add_argument("--max-steps", type=int, default=20000)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--grad-accum", type=int, default=4)
        p.add_argument("--lr", type=float, default=5e-5)
        p.add_argument("--warmup-steps", type=int, default=500)
        p.add_argument("--mixed-precision", default="bf16",
                       choices=["no", "fp16", "bf16"])
        p.add_argument("--conditioning-dropout", type=float, default=0.05)
        p.add_argument("--mask-floor", type=float, default=0.05)
        p.add_argument("--val-every", type=int, default=1000)
        p.add_argument("--checkpoint-every", type=int, default=2000)
        p.add_argument("--keep-checkpoints", type=int, default=2)
        p.add_argument("--gradient-checkpointing", action="store_true")
        p.add_argument("--resume", default=None)
        p.add_argument("--push-to-hub", default=None)

    for p in (run, infer):
        p.add_argument("--results", default="/workspace/results")
        p.add_argument("--model", default=None)
        p.add_argument("--steps", type=int, default=25)
        p.add_argument("--image-guidance", type=float, default=1.5)
        p.add_argument("--guidance", type=float, default=6.0)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--split", default="test", choices=["test", "val"])
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--no-results-tar", dest="results_tar", action="store_false")

    add_shared(train)
    train.add_argument("--workers", type=int, default=8)

    add_shared(infer)
    infer.add_argument("--output-dir", default="/workspace/coin-restorer-xl")

    add_shared(evaluate)
    evaluate.add_argument("--results", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--no-lpips", action="store_true")
    evaluate.add_argument("--no-fid", action="store_true")
    evaluate.add_argument("--train-log", default=None)

    add_shared(restore)
    restore.add_argument("--model", required=True)
    restore.add_argument("--input", required=True)
    restore.add_argument("--results", default="restored_out")
    restore.add_argument("--steps", type=int, default=30)
    restore.add_argument("--image-guidance", type=float, default=1.5)
    restore.add_argument("--guidance", type=float, default=6.0)
    restore.add_argument("--segment", action="store_true")

    run.add_argument("--skip-smoke", action="store_true",
                     help="skip the automatic smoke validation before the full run")
    run.add_argument("--no-sweep", action="store_true",
                     help="skip the automatic guidance sweep, use --guidance as is")
    infer.add_argument("--no-sweep", action="store_true", default=True,
                       help=argparse.SUPPRESS)

    return parser.parse_args()


# Dispatches the chosen subcommand to the pipeline.
def main():
    cfg = parse_args()
    pipeline = CoinRestorationPipeline(cfg)

    if cfg.command == "run":
        pipeline.run()
    elif cfg.command == "prepare":
        pipeline.prepare_data()
    elif cfg.command == "train":
        pipeline.train()
    elif cfg.command == "infer":
        pipeline.infer()
    elif cfg.command == "eval":
        pipeline.evaluate()
    elif cfg.command == "restore":
        pipeline.restore()


if __name__ == "__main__":
    main()
