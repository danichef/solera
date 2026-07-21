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

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

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


# Chip parameters, drawn independently of the sanding grade (the professor's
# point: a well-kept coin can still have a nicked rim).
# @params rng: numpy generator
# @output dict of ChipFilter overrides, or None for a chip-free coin
def sample_chip_params(rng):
    if rng.random() < CHIP_NONE_PROB:
        return None

    params = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in CHIP_RANGES.items()}
    params["big_chip_prob"] = 1.0 if rng.random() < BIG_CLIP_PROB else 0.0
    return params


# Wears one face down at the given grade, then adds independently sampled rim
# chips on top.
# @params image: float RGB face crop in [0, 1]
# @params coin_mask: float mask of the coin pixels
# @params sand_grade: one of SAND_GRADES
# @params seed: seed for this damage variant
# @output (damaged image, metadata record)
def damage_face(image, coin_mask, sand_grade, seed):
    rng = np.random.default_rng(seed)
    worn = SilverWearFilter.for_grade(sand_grade).apply(image, coin_mask, seed=seed)
    chips = sample_chip_params(rng)

    record = {
        "sand_grade": sand_grade,
        "seed": seed,
        "has_chips": chips is not None,
        "chip_amplitude": np.nan,
        "chip_threshold": np.nan,
        "chip_waviness": np.nan,
        "big_clip": False,
        "n_big_chips": 0,
    }
    if chips is None:
        return worn.image, record

    res = ChipFilter(**chips).apply(worn.image, worn.coin_mask, seed=seed + 1)
    big = res.params.get("big_chips") or []
    record["chip_amplitude"] = round(chips["amplitude"], 4)
    record["chip_threshold"] = round(chips["threshold"], 4)
    record["chip_waviness"] = round(chips["waviness"], 4)
    record["big_clip"] = chips["big_chip_prob"] > 0
    record["n_big_chips"] = len(big)
    return res.image, record


# Hashes a coin id into a split. Doing it by hash rather than shuffling means
# adding more coins later never shuffles an existing one into a different split.
# @params coin_id: filename stem of the scan
# @output "train", "val" or "test"
def stable_split(coin_id):
    bucket = int(hashlib.md5(coin_id.encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


# reproducible per-variant seed from whatever identifiers you pass
def stable_seed(*parts):
    key = "_".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# Pads an image (or mask) onto a square canvas and resizes to res. The fill is
# white for images, black for masks, so nothing bleeds in from the padding.
# @params img: float image or mask in [0, 1]
# @params res: output side in pixels
# @params fill: padding value
# @params resample: PIL resampling filter
# @output uint8 array, (res, res[, 3])
def to_square(img, res, fill, resample):
    h, w = img.shape[:2]
    side = max(h, w)
    if img.ndim == 3:
        canvas = np.full((side, side, img.shape[2]), fill, dtype=np.float32)
    else:
        canvas = np.full((side, side), fill, dtype=np.float32)

    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    out = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(out).resize((res, res), resample))


# save a float image as a square JPEG on white
def save_jpg(arr01, path, res):
    out = to_square(arr01, res, fill=1.0, resample=Image.LANCZOS)
    Image.fromarray(out).save(path, quality=95)


# save a coin mask as a square binary PNG
def save_mask(mask, path, res):
    out = to_square(mask, res, fill=0.0, resample=Image.BILINEAR)
    Image.fromarray(((out > 127) * 255).astype(np.uint8)).save(path)


# hands the shared config to each worker in the generation pool
def _init_worker(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg


# One coin scan, start to finish, inside a worker: segment the faces, save the
# clean target and its mask, then render all four damage grades. Runs in a pool
# so it has to catch its own errors and report them back rather than crash.
# @params jpg_path_str: path to the source scan
# @output ("ok", coin_id, rows) or ("error", coin_id, message)
def _process_coin(jpg_path_str):
    out = Path(_WORKER_CFG["out"])
    res = _WORKER_CFG["res"]
    jpg_path = Path(jpg_path_str)
    coin_id = jpg_path.stem
    split = _WORKER_CFG.get("force_split") or stable_split(coin_id)

    try:
        rgb = np.array(Image.open(jpg_path).convert("RGB"))
        faces = segment_faces(rgb)
        if not faces:
            return "error", coin_id, "no faces found"

        # a stray blob or two can slip through; keep the two biggest, left first
        if len(faces) > 2:
            faces = sorted(faces, key=lambda f: f.coin_mask.sum(), reverse=True)[:2]
            faces = sorted(faces, key=lambda f: f.bbox[0])

        rows = []
        for fi, face in enumerate(faces):
            clean_rel = f"{split}/clean/{coin_id}_{fi}.jpg"
            mask_rel = f"{split}/masks/{coin_id}_{fi}.png"
            save_jpg(face.image, out / clean_rel, res)
            save_mask(face.coin_mask, out / mask_rel, res)

            for grade in SAND_GRADES:
                seed = stable_seed(coin_id, fi, grade)
                damaged, record = damage_face(face.image, face.coin_mask, grade, seed)
                dmg_rel = f"{split}/damaged/{coin_id}_{fi}_{grade}.jpg"
                save_jpg(damaged, out / dmg_rel, res)

                rows.append(dict(record, coin_id=coin_id, face=fi, split=split,
                                 n_faces=len(faces), clean_path=clean_rel,
                                 mask_path=mask_rel, damaged_path=dmg_rel))

        return "ok", coin_id, rows

    except Exception as exc:
        return "error", coin_id, f"{type(exc).__name__}: {exc}"


# A quick eyeball sheet: one row per face, clean image beside its four grades.
# @params out: dataset root with the saved images
# @params rows: metadata rows to draw from
# @params n_faces: how many faces to show
def make_contact_sheet(out, rows, n_faces=6):
    groups, seen = [], set()
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
        imgs = [Image.open(out / face_rows[0]["clean_path"])]
        for grade in SAND_GRADES:
            match = next(r for r in face_rows if r["sand_grade"] == grade)
            imgs.append(Image.open(out / match["damaged_path"]))
        cells.append(imgs)

    if not cells:
        return

    cw = cells[0][0].width
    sheet = Image.new("RGB", (cw * 5, cw * len(cells)), (255, 255, 255))
    for r, imgs in enumerate(cells):
        for c, img in enumerate(imgs):
            sheet.paste(img, (c * cw, r * cw))
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
                           override_tar=False, override_samples=1)
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

        worker_cfg = {"out": str(out), "res": cfg.resolution,
                      "force_split": cfg.force_split}
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

    # Restores every image of one split with the trained model. With
    # --num-samples above 1 it draws several restorations per image from
    # different noise seeds, showing the distribution of plausible originals,
    # and composes a "fan" strip (damaged | samples | clean) per image.
    # The override arguments let the guidance sweep reuse this with different
    # settings without touching the main config.
    # @params override_split / override_limit / override_guidance /
    #         override_out / override_tar / override_samples: one-off settings
    # @output results folder: flat restored/+triptychs/ for one draw, or one
    #         by_coin/<coin>/ subfolder per image (samples + triptych + fan)
    def infer(self, override_split=None, override_limit=None,
              override_guidance=None, override_out=None, override_tar=None,
              override_samples=None):
        import torch
        from diffusers import DiffusionPipeline

        cfg = self.cfg
        data = Path(cfg.data)
        out = Path(override_out) if override_out else Path(cfg.results)
        split = override_split or cfg.split
        limit = override_limit if override_limit is not None else cfg.limit
        guidance = override_guidance if override_guidance is not None else cfg.guidance
        make_tar = override_tar if override_tar is not None else cfg.results_tar
        n_samples = override_samples if override_samples is not None else cfg.num_samples

        # With one draw the output stays flat; with several, each coin gets its
        # own subfolder holding all its samples, its triptych and its fan.
        multi = n_samples > 1
        if not multi:
            (out / "restored").mkdir(parents=True, exist_ok=True)
            (out / "triptychs").mkdir(exist_ok=True)
        else:
            (out / "by_coin").mkdir(parents=True, exist_ok=True)

        model_path = cfg.model or str(Path(cfg.output_dir) / "final")
        rows = self._metadata_rows(split)
        if limit:
            rows = rows[:limit]
        print(f"{len(rows)} {split} images x {n_samples} draws, "
              f"model: {model_path}, guidance {guidance}")

        pipe = DiffusionPipeline.from_pretrained(model_path,
                                                 torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)

        tasks = [(row, k) for row in rows for k in range(n_samples)]
        pending_fans = {}
        results = []
        bar = tqdm(range(0, len(tasks), cfg.batch), desc="restoring", unit="batch")
        for start in bar:
            chunk = tasks[start:start + cfg.batch]
            damaged = [Image.open(data / row["damaged_path"]).convert("RGB")
                       for row, _ in chunk]
            generators = [torch.Generator("cuda").manual_seed(int(row["seed"]) + 9973 * k)
                          for row, k in chunk]

            restored_batch = pipe([PROMPT] * len(chunk), image=damaged,
                                  height=cfg.resolution, width=cfg.resolution,
                                  num_inference_steps=cfg.steps,
                                  image_guidance_scale=cfg.image_guidance,
                                  guidance_scale=guidance,
                                  generator=generators).images

            for (row, k), damaged_img, restored in zip(chunk, damaged, restored_batch):
                base = f"{row['coin_id']}_{row['face']}_{row['sand_grade']}"
                if multi:
                    coin_dir = out / "by_coin" / base
                    coin_dir.mkdir(exist_ok=True)
                    restored_rel = f"by_coin/{base}/sample_{k}.png"
                    triptych_rel = f"by_coin/{base}/triptych.jpg"
                    fan_rel = f"by_coin/{base}/fan.jpg"
                else:
                    restored_rel = f"restored/{base}.png"
                    triptych_rel = f"triptychs/{base}.jpg"
                restored.save(out / restored_rel)

                record = dict(row, sample_idx=k, restored_path=restored_rel,
                              triptych_path=triptych_rel, steps=cfg.steps,
                              image_guidance=cfg.image_guidance,
                              guidance=guidance, model=model_path)

                clean = Image.open(data / row["clean_path"]).convert("RGB")
                if k == 0:
                    width, height = restored.size
                    triptych = Image.new("RGB", (width * 3, height), (255, 255, 255))
                    triptych.paste(damaged_img.resize((width, height)), (0, 0))
                    triptych.paste(restored, (width, 0))
                    triptych.paste(clean.resize((width, height)), (width * 2, 0))
                    triptych.save(out / triptych_rel, quality=92)

                if multi:
                    fan = pending_fans.setdefault(base, [None] * n_samples)
                    fan[k] = restored
                    if all(f is not None for f in fan):
                        self._save_fan(out / fan_rel, damaged_img, fan, clean)
                        del pending_fans[base]

                results.append(record)
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

    # Pastes a row of images into one strip and saves it.
    # @params path: destination jpg
    # @params images: PIL images, laid out left to right
    def _save_strip(self, path, *images):
        width, height = images[0].size
        strip = Image.new("RGB", (width * len(images), height), (255, 255, 255))
        for i, cell in enumerate(images):
            strip.paste(cell.resize((width, height)), (i * width, 0))
        strip.save(path, quality=90)

    # Lays out one image's sample fan: the damaged input, every drawn
    # restoration, and the clean target in a single row.
    # @params path: destination jpg
    # @params damaged_img: the conditioning input
    # @params samples: list of restored PIL images
    # @params clean: the ground-truth image
    def _save_fan(self, path, damaged_img, samples, clean):
        self._save_strip(path, damaged_img, *samples, clean)

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

        # --replot reuses the already scored csv, so plots and reports can be
        # regenerated (or extended) without touching a single image again.
        per_image_csv = results / "results_per_image.csv"
        if cfg.replot and per_image_csv.exists():
            full = pd.read_csv(per_image_csv)
            for column in ("has_chips", "big_clip"):
                full[column] = full[column].astype(str).str.lower() == "true"
            print(f"replotting {len(full)} previously scored results")
        else:
            full = self._score_results(data, results, res, structural_similarity)
            full.to_csv(per_image_csv, index=False)

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

        best_of = self._best_of_samples(full)
        if best_of is not None:
            print(f"\n== best-of-{best_of['n']} across samples ==")
            for key in ("psnr", "ssim", "lpips"):
                if key in best_of:
                    print(f"  {key}: {best_of[key]}")

        fid = None if cfg.no_fid else self._compute_fid(full)

        multi = best_of is not None
        if multi:
            self._make_plots(self._collapse_per_image(full, "best"), results,
                             "plots/best")
            self._make_plots(self._collapse_per_image(full, "average"), results,
                             "plots/average")
            self._make_generation_plots(full, results)
        else:
            self._make_plots(full, results, "plots")
        self._write_report(full, by_grade, by_chip, by_clip, fid, best_of,
                           multi, results)

        print(f"\n-> {results / 'results_per_image.csv'}")
        print(f"-> {results / 'results_summary.csv'}")
        print(f"-> {results / 'results_report.md'}")
        print(f"-> {results / 'plots/'}")

    # Scores every restored image against its clean target with masked
    # PSNR / SSIM / LPIPS, plus the damaged-input baseline.
    # @params data: dataset folder
    # @params results: results folder holding results_metadata.csv
    # @params res: evaluation resolution
    # @params structural_similarity: skimage ssim function
    # @output per-image dataframe of metadata plus metric columns
    def _score_results(self, data, results, res, structural_similarity):
        import pandas as pd

        cfg = self.cfg
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

        return pd.concat([df, pd.DataFrame(records)], axis=1)

    # When several samples were drawn per image, reports the oracle score of
    # the best draw: how good the distribution's best answer is, per image,
    # averaged over the test set.
    # @params full: per-image dataframe with metric columns
    # @output dict of formatted best-of-N stats, or None for single samples
    def _best_of_samples(self, full):
        if "sample_idx" not in full.columns or full["sample_idx"].nunique() < 2:
            return None

        keys = ["coin_id", "face", "sand_grade"]
        result = {"n": int(full["sample_idx"].nunique())}
        directions = [("psnr", "max"), ("ssim", "max"), ("lpips", "min")]
        for metric, how in directions:
            if metric not in full.columns:
                continue
            per_image = full.groupby(keys)[metric].agg(how)
            result[metric] = f"{per_image.mean():.4f} ± {per_image.std():.4f}"
        return result

    # Computes distribution-level FID: restored vs clean, plus the damaged vs
    # clean baseline that shows how far restoration moved the distribution.
    # Gathers restored images from the metadata so it works whether they sit
    # in a flat folder or in per-coin subfolders.
    # @params full: per-image dataframe with the restored_path column
    # @output dict with both scores, or None when pytorch-fid is missing
    def _compute_fid(self, full):
        try:
            from pytorch_fid import fid_score
        except ImportError:
            print("pytorch-fid not installed, skipping FID (pip install pytorch-fid)")
            return None

        import tempfile
        cfg = self.cfg
        clean_dir = str(Path(cfg.data) / "test" / "clean")
        damaged_dir = str(Path(cfg.data) / "test" / "damaged")

        # A flat directory of all restored images, with unique names, that FID
        # can scan regardless of the on-disk layout.
        flat = Path(tempfile.mkdtemp(prefix="fid_restored_"))
        for rel in full["restored_path"]:
            src = (Path(cfg.results) / rel).resolve()
            (flat / rel.replace("/", "_")).symlink_to(src)

        try:
            print("computing FID (restored vs clean)")
            fid_restored = fid_score.calculate_fid_given_paths(
                [str(flat), clean_dir], batch_size=32, device=cfg.device, dims=2048)
            print("computing FID baseline (damaged vs clean)")
            fid_damaged = fid_score.calculate_fid_given_paths(
                [damaged_dir, clean_dir], batch_size=32, device=cfg.device, dims=2048)
        finally:
            shutil.rmtree(flat)

        print(f"FID restored: {fid_restored:.2f}   FID damaged: {fid_damaged:.2f}")
        return {"restored": fid_restored, "damaged": fid_damaged}

    # Collapses several samples per image down to one row each, so the plots
    # can show either the best draw or the average draw.
    # @params full: per-image, per-sample dataframe
    # @params how: "best" (oracle per metric) or "average" (mean over draws)
    # @output one-row-per-image dataframe
    def _collapse_per_image(self, full, how):
        keys = ["coin_id", "face", "sand_grade"]
        best_dir = {"psnr": "max", "ssim": "max", "lpips": "min",
                    "psnr_unmasked": "max", "psnr_damaged": "mean",
                    "lpips_damaged": "mean"}
        agg = {}
        for col in full.columns:
            if col in best_dir:
                agg[col] = best_dir[col] if how == "best" else "mean"
            elif col in ("chip_bin", "big_clip"):
                agg[col] = "first"
        return full.groupby(keys, as_index=False).agg(agg)

    # Draws the evaluation figures: metrics by damage state, the per-image
    # improvement scatter, metric distributions, and the training loss curve.
    # @params full: per-image dataframe with metric columns
    # @params results: results folder
    # @params subdir: where the figures land, e.g. "plots" or "plots/best"
    def _make_plots(self, full, results, subdir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = Path(results) / subdir
        plots.mkdir(parents=True, exist_ok=True)

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

    # Draws the figures only a multi-draw run enables: the full metric
    # distribution over every generation (not collapsed to best or mean),
    # split by damage grade, and the per-image spread across draws, which
    # shows how much the model's answers disagree for the same input.
    # @params full: per-image, per-sample dataframe
    # @params results: results folder
    def _make_generation_plots(self, full, results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = Path(results) / "plots/generations"
        plots.mkdir(parents=True, exist_ok=True)

        grade_colors = {"f": "#2a78d6", "vg": "#1baf7a",
                        "g": "#eda100", "ag": "#4a3aa7"}

        def style(ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
            ax.set_axisbelow(True)

        metric_titles = [("psnr", "PSNR (dB, higher better)"),
                         ("ssim", "SSIM (higher better)"),
                         ("lpips", "LPIPS (lower better)")]
        metrics = [(m, t) for m, t in metric_titles if m in full.columns]
        grades = [g for g in SAND_GRADES if (full["sand_grade"] == g).any()]
        n_draws = int(full["sample_idx"].nunique())

        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
        for ax, (metric, title) in zip(np.atleast_1d(axes), metrics):
            data = [full.loc[full["sand_grade"] == g, metric].dropna() for g in grades]
            parts = ax.violinplot(data, showmedians=True, widths=0.8)
            for body, grade in zip(parts["bodies"], grades):
                body.set_facecolor(grade_colors[grade])
                body.set_alpha(0.75)
                body.set_edgecolor("none")
            for piece in ("cmedians", "cbars", "cmins", "cmaxes"):
                parts[piece].set_color("#52514e")
                parts[piece].set_linewidth(1)
            ax.set_xticks(range(1, len(grades) + 1))
            ax.set_xticklabels(grades)
            ax.set_title(f"{title}\nall {n_draws} draws, by sanding grade",
                         fontsize=11)
            style(ax)
        fig.tight_layout()
        fig.savefig(plots / "metric_violins.png", bbox_inches="tight")
        plt.close(fig)

        keys = ["coin_id", "face", "sand_grade"]
        spread = full.groupby(keys)[[m for m, _ in metrics]].std()
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
        for ax, (metric, title) in zip(np.atleast_1d(axes), metrics):
            ax.hist(spread[metric].dropna(), bins=40, color="#2a78d6")
            ax.set_xlabel(f"std of {metric} across {n_draws} draws")
            ax.set_ylabel("images")
            style(ax)
        fig.suptitle("Per-image spread across generations", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(plots / "per_image_spread.png", bbox_inches="tight")
        plt.close(fig)

    # Writes a human-readable evaluation report next to the csv outputs.
    # @params full: per-image dataframe
    # @params by_grade / by_chip / by_clip: grouped summary tables
    # @params fid: FID scores dict or None
    # @params best_of: best-of-N sample stats dict or None
    # @params multi: whether several samples were drawn per image
    # @params results: results folder
    def _write_report(self, full, by_grade, by_chip, by_clip, fid, best_of,
                      multi, results):
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

        if best_of:
            lines.append(f"## Best of {best_of['n']} samples per image")
            lines.append("")
            lines.append("Oracle score of the best draw per image, showing how "
                         "good an answer the sampled distribution contains.")
            lines.append("")
            lines.append("| metric | best-of-N |")
            lines.append("|---|---|")
            for key in ("psnr", "ssim", "lpips"):
                if key in best_of:
                    lines.append(f"| {key.upper()} | {best_of[key]} |")
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
        if multi:
            lines.append("- `plots/best/` — figures using each image's best draw")
            lines.append("- `plots/average/` — figures using the mean over draws")
            lines.append("- `plots/generations/` — full metric distributions "
                         "over every draw, and the per-image spread")
            lines.append("- `by_coin/<coin>/` — every sample, triptych and fan "
                         "strip for each coin")
        else:
            lines.append("- `plots/metrics_by_grade.png`, `plots/metrics_by_chip.png` "
                         "— restored vs damaged by damage state")
            lines.append("- `plots/lpips_improvement.png` — per-image before/after")
            lines.append("- `plots/distributions.png` — metric histograms")
        if self.cfg.train_log and Path(self.cfg.train_log).exists():
            base = "plots/best" if multi else "plots"
            lines.append(f"- `{base}/training_loss.png` — smoothed loss curve")

        (Path(results) / "results_report.md").write_text("\n".join(lines) + "\n")

    # Restores arbitrary coin images (e.g. a folder of scans) with a saved
    # model. With --num-samples above 1 each coin gets its own subfolder
    # holding every drawn restoration plus a fan strip (input | samples).
    # @output restored PNGs in cfg.results, one subfolder per coin when multi
    def restore(self):
        import torch
        from diffusers import DiffusionPipeline

        cfg = self.cfg
        device, dtype = self._pick_device()
        n_samples = cfg.num_samples
        multi = n_samples > 1
        print(f"device={device} dtype={dtype}, {n_samples} draws per coin")

        pipe = DiffusionPipeline.from_pretrained(cfg.model, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_vae_slicing()

        source = Path(cfg.input)
        if source.is_dir():
            paths = sorted(p for p in source.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        else:
            paths = [source]

        out = Path(cfg.results)
        out.mkdir(parents=True, exist_ok=True)

        for path in tqdm(paths, desc="restoring", unit="coin"):
            for suffix, img in self._input_faces(path):
                name = f"{path.stem}{suffix}"
                coin_dir = out / name
                flat_png = out / f"{name}_restored.png"

                # already done on a previous run, skip it
                if multi and (coin_dir / "fan.jpg").exists():
                    continue
                if not multi and flat_png.exists():
                    continue

                # all draws in one batched call, one generator per seed
                generators = [torch.Generator(device).manual_seed(cfg.seed + 9973 * k)
                              for k in range(n_samples)]
                samples = pipe([PROMPT] * n_samples, image=[img] * n_samples,
                               height=cfg.resolution, width=cfg.resolution,
                               num_inference_steps=cfg.steps,
                               image_guidance_scale=cfg.image_guidance,
                               guidance_scale=cfg.guidance,
                               generator=generators).images

                if multi:
                    coin_dir.mkdir(exist_ok=True)
                    for k, restored in enumerate(samples):
                        restored.save(coin_dir / f"sample_{k}.png")
                    self._save_strip(coin_dir / "fan.jpg", img, *samples)
                else:
                    samples[0].save(flat_png)

    # Measures train/test leakage between two folders of images. Every image
    # is embedded with an ImageNet ResNet-50, each test image is matched to
    # its nearest training neighbours by cosine similarity, and the top
    # matches are re-scored with SSIM (and LPIPS when installed) at pixel
    # level. A tall isolated peak in the similarity profile means a test coin
    # has a near-duplicate sitting in the training set.
    #
    # With --results (an eval folder) the test set is first narrowed to the
    # best, worst and an even spread of middle coins by mean restoration score
    # over their generations, so the pixel scoring and montages run on a
    # representative subset rather than every test coin. The test-vs-train
    # cosine is a single matmul over cached embeddings either way, so the
    # subset saves the scoring and rendering, not the search.
    # @output leakage_pairs.csv, leakage_report.md, plots/ and either
    #         top_pairs.jpg (full) or per-tier montages + a quality-vs-leakage
    #         scatter (eval-guided) in cfg.out
    def leakage(self):
        import pandas as pd
        import torch
        from skimage.metrics import structural_similarity

        cfg = self.cfg
        device = cfg.device or self._pick_device()[0]
        out = Path(cfg.out)
        (out / "plots").mkdir(parents=True, exist_ok=True)

        embed = self._embedder(device)

        selection = tier_of = None
        if cfg.results:
            selection = self._select_by_eval(cfg.results, cfg.rank_metric,
                                             cfg.group_size)
            tier_of = {s["coin_id"]: s for s in selection}
            test_paths = self._paths_for_coins(cfg.test_dir, selection)
            test_records, test_emb = self._embed_folder(
                embed, "embedding test", paths=test_paths)
        else:
            test_records, test_emb = self._embed_folder(
                embed, "embedding test", folder=cfg.test_dir)

        # Training embeddings are cached to disk keyed by the folder's
        # contents: the 8k-image forward pass is the real cost here and it is
        # identical across reruns, so it is paid once.
        train_records, train_emb = self._embed_folder(
            embed, "embedding train", folder=cfg.train_dir,
            cache_dir=out / "cache")
        print(f"{len(test_records)} test vs {len(train_records)} train images, "
              f"device {device}")

        # cosine top-k, chunked over test rows so the full similarity matrix
        # never has to exist at once
        k = min(cfg.topk, len(train_records))
        scores, indices = [], []
        for start in range(0, len(test_emb), 512):
            sims = test_emb[start:start + 512] @ train_emb.T
            top = sims.topk(k, dim=1)
            scores.append(top.values)
            indices.append(top.indices)
        scores = torch.cat(scores).tolist()
        indices = torch.cat(indices).tolist()

        lpips_scorer = None
        if not cfg.no_lpips:
            try:
                lpips_scorer = self._lpips_scorer(device)
            except ImportError:
                print("lpips not installed, skipping (pip install lpips)")

        ones = np.ones((cfg.resolution, cfg.resolution), dtype=np.float32)
        rows = []
        for t, (name, path, face_idx) in enumerate(tqdm(test_records,
                                                        desc="scoring",
                                                        unit="img")):
            test01 = self._face01(path, face_idx)
            for rank in range(k):
                j = indices[t][rank]
                train_name, train_path, train_face = train_records[j]
                train01 = self._face01(train_path, train_face)
                row = {
                    "test_image": name,
                    "coin_id": path.stem,
                    "rank": rank + 1,
                    "train_image": train_name,
                    "cosine": scores[t][rank],
                    "ssim": self._masked_ssim(test01, train01, ones,
                                              structural_similarity),
                }
                if lpips_scorer:
                    row["lpips"] = lpips_scorer(test01, train01, ones)
                if tier_of is not None:
                    info = tier_of.get(path.stem, {})
                    row["tier"] = info.get("tier")
                    row["rank_score"] = info.get("score")
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out / "leakage_pairs.csv", index=False)

        if selection is None:
            order = sorted(range(len(test_records)),
                           key=lambda t: -scores[t][0])[:cfg.pairs]
            pairs = [(f"{test_records[t][0]}  |  "
                      f"{train_records[indices[t][0]][0]}   "
                      f"cos {scores[t][0]:.3f}",
                      test_records[t], train_records[indices[t][0]])
                     for t in order]
            self._pair_sheet(pairs, out / "top_pairs.jpg")
        else:
            self._leakage_eval_montages(test_records, train_records, scores,
                                        indices, tier_of, cfg.rank_metric, out)

        self._leakage_plots(df, out)
        self._write_leakage_report(df, len(test_records), len(train_records),
                                   selection, cfg.rank_metric, out)

        print(f"\n-> {out / 'leakage_pairs.csv'}")
        print(f"-> {out / 'leakage_report.md'}")
        if selection is None:
            print(f"-> {out / 'top_pairs.jpg'}")
        else:
            print(f"-> {out / 'montage_best.jpg'} (+ middle, worst)")
            print(f"-> {out / 'plots' / 'quality_vs_leakage.png'}")
        print(f"-> {out / 'plots/'}")

    # Ranks the evaluated test coins by their mean restoration metric over all
    # generations and returns three tiers: the best group-size coins, the
    # worst group-size, and group-size sampled at even rank intervals across
    # the middle band, so the middle is a spread rather than a median cluster.
    # @params results_dir: eval folder holding results_per_image.csv
    # @params metric: one of lpips / psnr / ssim
    # @params n: coins per tier
    # @output list of {coin_id, tier, score}, best tier first
    def _select_by_eval(self, results_dir, metric, n):
        import pandas as pd

        csv_path = Path(results_dir) / "results_per_image.csv"
        if not csv_path.exists():
            sys.exit(f"no results_per_image.csv in {results_dir} "
                     "(run 'pipeline.py eval' first)")
        df = pd.read_csv(csv_path, dtype={"coin_id": str})
        if metric not in df.columns:
            have = [c for c in ("lpips", "psnr", "ssim") if c in df.columns]
            sys.exit(f"metric '{metric}' not in results_per_image.csv; "
                     f"available: {have or 'none'}")

        per_coin = df.groupby("coin_id")[metric].mean().dropna()
        lower_better = metric == "lpips"
        ranked = per_coin.sort_values(ascending=lower_better)  # best first
        ordered = list(ranked.index)
        n = n if len(ordered) >= 3 * n else len(ordered) // 3
        if n < 1:
            sys.exit(f"only {len(ordered)} coins scored; too few to form tiers")

        best = ordered[:n]
        worst = ordered[len(ordered) - n:]
        band = ordered[n:len(ordered) - n]
        if band:
            picks = np.linspace(0, len(band) - 1, min(n, len(band)))
            middle = list(dict.fromkeys(band[int(round(i))] for i in picks))
        else:
            middle = []

        selection = []
        for tier, ids in (("best", best), ("middle", middle), ("worst", worst)):
            for coin_id in ids:
                selection.append({"coin_id": str(coin_id), "tier": tier,
                                  "score": float(ranked[coin_id])})
        print(f"eval-guided: {len(best)} best / {len(middle)} middle / "
              f"{len(worst)} worst coins by mean {metric} "
              f"(of {len(ordered)} scored)")
        return selection

    # Maps each selected coin_id to its scan in the test folder, matching on
    # filename stem.
    # @params test_dir: folder of test scans
    # @params selection: list from _select_by_eval
    # @output list of Paths, in selection order, for the coins that were found
    def _paths_for_coins(self, test_dir, selection):
        test_dir = Path(test_dir)
        by_stem = {}
        for path in sorted(test_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                by_stem.setdefault(path.stem, path)

        paths, missing = [], []
        for item in selection:
            path = by_stem.get(item["coin_id"])
            if path is not None:
                paths.append(path)
            else:
                missing.append(item["coin_id"])
        if missing:
            print(f"warning: {len(missing)} selected coins missing from "
                  f"{test_dir} (e.g. {missing[:3]})")
        if not paths:
            sys.exit(f"none of the selected coins were found in {test_dir}")
        return paths

    # Renders one montage per quality tier (each coin beside its closest
    # training match, labelled with its score and cosine) plus the scatter of
    # restoration score against nearest-neighbour cosine.
    # @params test_records / train_records: (name, path, face_idx) lists
    # @params scores / indices: per-test top-k cosines and train indices
    # @params tier_of: coin_id -> {tier, score} from _select_by_eval
    # @params metric: ranking metric name, for labels
    # @params out: output folder
    def _leakage_eval_montages(self, test_records, train_records, scores,
                               indices, tier_of, metric, out):
        # one representative pair per coin: the face whose closest training
        # neighbour is nearest (coins may contribute two faces under --segment)
        best = {}
        for t, rec in enumerate(test_records):
            coin_id = rec[1].stem
            if coin_id not in tier_of:
                continue
            cosine = scores[t][0]
            if coin_id not in best or cosine > best[coin_id][0]:
                best[coin_id] = (cosine, rec, train_records[indices[t][0]])

        lower_better = metric == "lpips"
        points, tiers = [], {"best": [], "middle": [], "worst": []}
        for coin_id, (cosine, test_rec, train_rec) in best.items():
            info = tier_of[coin_id]
            points.append({"tier": info["tier"], "score": info["score"],
                           "cosine": cosine})
            tiers[info["tier"]].append((info["score"], cosine, test_rec,
                                        train_rec))

        for tier, items in tiers.items():
            if not items:
                continue
            items.sort(key=lambda it: it[0], reverse=not lower_better)
            sheet = [(f"{test_rec[1].stem}   {metric} {score:.3f}   "
                      f"cos {cosine:.3f}", test_rec, train_rec)
                     for score, cosine, test_rec, train_rec in items]
            self._pair_sheet(sheet, out / f"montage_{tier}.jpg",
                             title=f"{tier.upper()} restoration tier - "
                                   f"{len(items)} coins, ranked by mean {metric}")

        self._leakage_eval_scatter(points, metric, out)

    # Scatter of each selected coin's restoration score against its
    # nearest-neighbour cosine, coloured by tier: if the best-restored coins
    # cluster at high cosine, restoration quality is tracking leakage.
    # @params points: list of {tier, score, cosine}
    # @params metric: ranking metric name, for the axis label
    # @params out: output folder
    def _leakage_eval_scatter(self, points, metric, out):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tier_colors = {"best": "#1baf7a", "middle": "#eda100", "worst": "#d6452a"}
        plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                             "axes.edgecolor": "#d5d4cf"})
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        for tier in ("best", "middle", "worst"):
            xs = [p["cosine"] for p in points if p["tier"] == tier]
            ys = [p["score"] for p in points if p["tier"] == tier]
            ax.scatter(xs, ys, s=30, alpha=0.85, linewidths=0,
                       color=tier_colors[tier], label=tier)
        direction = "lower better" if metric == "lpips" else "higher better"
        ax.set_xlabel("nearest-neighbour cosine similarity (test vs train)")
        ax.set_ylabel(f"mean {metric} over generations ({direction})")
        ax.set_title("Restoration quality vs training-set leakage", fontsize=11)
        ax.legend(title="quality tier", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(out / "plots" / "quality_vs_leakage.png", bbox_inches="tight")
        plt.close(fig)

    # Builds the embedding function: an ImageNet ResNet-50 with the
    # classifier removed, so each image maps to a 2048-d feature whose cosine
    # similarity acts as a perceptual nearness score.
    # @params device: torch device string
    # @output callable (list of PIL images) -> L2-normalized cpu tensor (n, 2048)
    def _embedder(self, device):
        import torch
        from torchvision import models

        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        net.eval().to(device)

        # manual ImageNet preprocessing from PIL bytes, so it also works on
        # torch builds whose numpy interop is broken
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def prep(img):
            img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
            flat = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
            pixels = flat.view(224, 224, 3).permute(2, 0, 1).float() / 255.0
            return (pixels - mean) / std

        def embed(images):
            batch = torch.stack([prep(img) for img in images]).to(device)
            with torch.no_grad():
                features = net(batch)
            return torch.nn.functional.normalize(features, dim=1).cpu()

        return embed

    # Embeds a set of images (every segmented face of each when --segment is
    # set), optionally serving from and writing to an on-disk cache. Pass
    # either a folder to scan or an explicit list of paths.
    # @params embed: batch embedding function from _embedder
    # @params desc: progress bar label
    # @params folder: directory of scans (scanned when paths is None)
    # @params paths: explicit list of image paths to embed instead
    # @params cache_dir: when set, reuse/store embeddings keyed by file content
    # @output (records, embeddings) where records are (name, path, face_idx)
    def _embed_folder(self, embed, desc, folder=None, paths=None,
                      cache_dir=None):
        import torch

        if paths is None:
            paths = sorted(p for p in Path(folder).iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            sys.exit(f"no images found in {folder}")

        # Serialised with torch.save (not numpy) so the cache works on the
        # same builds with broken numpy<->torch interop that _embedder already
        # sidesteps; the record names/paths/faces ride alongside as JSON.
        cache_file = cache_meta = None
        if cache_dir is not None:
            key = self._emb_cache_key(paths)
            cache_dir = Path(cache_dir)
            cache_file = cache_dir / f"emb_{key}.pt"
            cache_meta = cache_dir / f"emb_{key}.json"
            if cache_file.exists() and cache_meta.exists():
                meta = json.loads(cache_meta.read_text())
                records = [(n, Path(p), int(f)) for n, p, f in
                           zip(meta["names"], meta["paths"], meta["faces"])]
                print(f"{desc}: {len(records)} cached embeddings "
                      f"({cache_file.name})")
                return records, torch.load(cache_file, map_location="cpu")

        records, chunks, pending = [], [], []
        for path in tqdm(paths, desc=desc, unit="img"):
            for face_idx, (suffix, img) in enumerate(self._input_faces(path)):
                records.append((f"{path.stem}{suffix}", path, face_idx))
                pending.append(img)
                if len(pending) == self.cfg.batch:
                    chunks.append(embed(pending))
                    pending = []
        if pending:
            chunks.append(embed(pending))
        emb = torch.cat(chunks)

        if cache_file is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(emb, cache_file)
            cache_meta.write_text(json.dumps({
                "names": [r[0] for r in records],
                "paths": [str(r[1]) for r in records],
                "faces": [r[2] for r in records]}))
            print(f"{desc}: cached -> {cache_file}")
        return records, emb

    # Content key for the embedding cache: the segment flag, the folder the
    # images live in, and each file's name, size and mtime, so any edit to the
    # set (or moving it) recomputes rather than serving stale vectors.
    # @params paths: the image paths being embedded
    # @output short hex digest
    def _emb_cache_key(self, paths):
        digest = hashlib.md5()
        digest.update(f"seg={int(self.cfg.segment)}".encode())
        digest.update(str(paths[0].parent.resolve()).encode())
        for path in paths:
            info = path.stat()
            digest.update(
                f"{path.name}:{info.st_size}:{int(info.st_mtime)}".encode())
        return digest.hexdigest()[:16]

    # Reloads one prepared face as a float array for pixel-level scoring.
    # @params path: source image
    # @params face_idx: which face of the image
    # @output float array (res, res, 3) in [0, 1]
    def _face01(self, path, face_idx):
        img = self._input_faces(path)[face_idx][1]
        return np.asarray(img, dtype=np.float32) / 255.0

    # Renders labelled test/train pairs two-per-row into one sheet, each test
    # face beside its matched training face, so suspected leaks can be checked
    # by eye.
    # @params items: list of (label, test record, train record)
    # @params path: destination jpg
    # @params title: optional heading drawn across the top
    def _pair_sheet(self, items, path, title=None):
        from PIL import ImageDraw

        cell = 320
        label_height = 26
        cols = 2
        top_pad = 34 if title else 0
        rows = (len(items) + cols - 1) // cols
        width = cols * cell * 2 + (cols - 1) * 16
        height = top_pad + rows * (cell + label_height)
        sheet = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        if title:
            draw.text((6, 10), title, fill=(20, 20, 20))
        for i, (label, test_rec, train_rec) in enumerate(items):
            r, c = divmod(i, cols)
            x = c * (cell * 2 + 16)
            y = top_pad + r * (cell + label_height)
            test_img = self._input_faces(test_rec[1])[test_rec[2]][1]
            train_img = self._input_faces(train_rec[1])[train_rec[2]][1]
            sheet.paste(test_img.resize((cell, cell)), (x, y + label_height))
            sheet.paste(train_img.resize((cell, cell)), (x + cell, y + label_height))
            draw.text((x + 4, y + 7), label, fill=(20, 20, 20))
        sheet.save(path, quality=90)

    # Draws the leakage figures: the nearest-neighbour similarity histogram,
    # the sorted similarity profile, and cosine against SSIM for every top-1
    # match.
    # @params df: long-format pairs dataframe with a rank column
    # @params out: output folder
    def _leakage_plots(self, df, out):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        accent = "#2a78d6"
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

        top1 = df[df["rank"] == 1]

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
        axes[0].hist(top1["cosine"], bins=40, color=accent)
        axes[0].set_xlabel("nearest-neighbour cosine similarity")
        axes[0].set_ylabel("test images")
        axes[0].set_title("How close is each test image\nto its nearest "
                          "training image?", fontsize=11)
        profile = np.sort(top1["cosine"].to_numpy())
        axes[1].plot(np.linspace(0, 100, len(profile)), profile,
                     color=accent, linewidth=1.6)
        axes[1].set_xlabel("percentile of test images")
        axes[1].set_ylabel("cosine similarity")
        axes[1].set_title("Similarity profile\n(a spike at the right = "
                          "near-duplicates)", fontsize=11)
        for ax in axes:
            style(ax)
        fig.tight_layout()
        fig.savefig(out / "plots/nn_similarity.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        ax.scatter(top1["cosine"], top1["ssim"], s=10, alpha=0.6,
                   color=accent, linewidths=0)
        ax.set_xlabel("embedding cosine similarity")
        ax.set_ylabel("SSIM of the matched pair")
        ax.set_title("Perceptual vs pixel similarity of each\ntest image's "
                     "closest training match", fontsize=11)
        style(ax)
        ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(out / "plots/cosine_vs_ssim.png", bbox_inches="tight")
        plt.close(fig)

    # Writes the human-readable leakage summary.
    # @params df: long-format pairs dataframe
    # @params n_test / n_train: image counts per folder
    # @params selection: eval-guided tiers, or None for the full run
    # @params metric: ranking metric name (used only in eval-guided mode)
    # @params out: output folder
    def _write_leakage_report(self, df, n_test, n_train, selection, metric, out):
        top1 = df[df["rank"] == 1]
        columns = [("cosine", "max"), ("ssim", "max")]
        if "lpips" in df.columns:
            columns.append(("lpips", "min"))

        lines = ["# Leakage report", ""]
        lines.append(f"{n_test} test images matched against {n_train} "
                     "training images.")
        lines.append("Images are embedded with an ImageNet ResNet-50 and each "
                     "test image is paired with its nearest training image by "
                     "cosine similarity; SSIM and LPIPS then score the matched "
                     "pair at pixel level. LPIPS is a distance, so lower means "
                     "more similar.")
        lines.append("")
        if selection is not None:
            counts = {t: sum(1 for s in selection if s["tier"] == t)
                      for t in ("best", "middle", "worst")}
            lines.append("## Eval-guided subset")
            lines.append("")
            lines.append(f"Leakage was measured only for the coins ranked by "
                         f"mean **{metric}** over their generations: the best "
                         "and worst tiers plus an even spread of the middle.")
            lines.append("")
            lines.append("| tier | coins | mean NN cosine |")
            lines.append("|---|---|---|")
            for tier in ("best", "middle", "worst"):
                sub = (top1[top1["tier"] == tier] if "tier" in top1.columns
                       else top1.iloc[0:0])
                cos = f"{sub['cosine'].mean():.4f}" if len(sub) else "n/a"
                lines.append(f"| {tier} | {counts[tier]} | {cos} |")
            lines.append("")
            lines.append("See `montage_best.jpg` / `montage_middle.jpg` / "
                         "`montage_worst.jpg` and "
                         "`plots/quality_vs_leakage.png`.")
            lines.append("")
        lines.append("## Nearest-neighbour similarity")
        lines.append("")
        header = " | ".join(name for name, _ in columns)
        lines.append(f"| stat | {header} |")
        lines.append("|---|" + "---|" * len(columns))
        stats = [("mean", "mean"), ("median", "median"),
                 ("most similar", None)]
        for label, how in stats:
            cells = []
            for name, extreme in columns:
                value = getattr(top1[name], how or extreme)()
                cells.append(f"{value:.4f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("## Ten most similar pairs")
        lines.append("")
        lines.append("| test | train | " + header + " |")
        lines.append("|---|---|" + "---|" * len(columns))
        closest = top1.sort_values("cosine", ascending=False).head(10)
        for _, row in closest.iterrows():
            cells = " | ".join(f"{row[name]:.4f}" for name, _ in columns)
            lines.append(f"| {row['test_image']} | {row['train_image']} "
                         f"| {cells} |")
        lines.append("")
        lines.append("Cosine above ~0.95 together with high SSIM usually means "
                     "the same coin appears in both folders. Check those pairs "
                     "in `top_pairs.jpg`.")
        (out / "leakage_report.md").write_text("\n".join(lines) + "\n")

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
    leakage = sub.add_parser("leakage", help="nearest-neighbour similarity "
                             "between two image folders (e.g. test vs "
                             "training) to measure dataset leakage")

    for p in (run, prepare):
        add_shared(p)
        p.add_argument("--src", default=str(Path.home() / "Downloads/data_part_1"))
        p.add_argument("--num-coins", type=int, default=9999)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--tar", action="store_true",
                       help="also pack the dataset into dataset.tar for upload")
        p.add_argument("--force-split", default=None,
                       choices=["train", "val", "test"],
                       help="put every coin in this split instead of the "
                            "hashed 80/10/10 assignment (e.g. to build a "
                            "held-out test set from a folder of scans)")

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
        p.add_argument("--num-samples", type=int, default=1,
                       help="restorations drawn per image; above 1 also writes "
                            "fans/ strips showing the distribution")
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
    evaluate.add_argument("--replot", action="store_true",
                          help="rebuild plots and reports from an existing "
                               "results_per_image.csv without rescoring")

    leakage.add_argument("--test-dir", required=True)
    leakage.add_argument("--train-dir", required=True)
    leakage.add_argument("--out", default="leakage_out")
    leakage.add_argument("--segment", action="store_true",
                         help="segment each scan into faces and compare "
                              "per face instead of per full image")
    leakage.add_argument("--topk", type=int, default=5,
                         help="training neighbours kept per test image")
    leakage.add_argument("--pairs", type=int, default=20,
                         help="most similar pairs rendered in top_pairs.jpg")
    leakage.add_argument("--batch", type=int, default=32)
    leakage.add_argument("--resolution", type=int, default=512,
                         help="side at which faces are squared for pixel "
                              "scoring and the pair sheet")
    leakage.add_argument("--device", default=None,
                         help="cuda / mps / cpu, autodetected when omitted")
    leakage.add_argument("--no-lpips", action="store_true")
    leakage.add_argument("--results", default=None,
                         help="eval results folder holding "
                              "results_per_image.csv; enables the eval-guided "
                              "subset that measures leakage only for the best, "
                              "worst and middle coins by restoration score")
    leakage.add_argument("--rank-metric", default="lpips",
                         choices=["lpips", "psnr", "ssim"],
                         help="restoration metric that ranks coins into tiers "
                              "in --results mode (default lpips, lower better)")
    leakage.add_argument("--group-size", type=int, default=20,
                         help="coins per tier (best / middle / worst) in "
                              "--results mode")

    add_shared(restore)
    restore.add_argument("--model", required=True)
    restore.add_argument("--input", required=True)
    restore.add_argument("--results", default="restored_out")
    restore.add_argument("--steps", type=int, default=30)
    restore.add_argument("--image-guidance", type=float, default=1.5)
    restore.add_argument("--guidance", type=float, default=6.0)
    restore.add_argument("--segment", action="store_true")
    restore.add_argument("--num-samples", type=int, default=1,
                         help="restorations drawn per coin; above 1 writes one "
                              "subfolder per coin with all samples and a fan")

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
    elif cfg.command == "leakage":
        pipeline.leakage()


if __name__ == "__main__":
    main()
