"""Rebuild results_per_image.csv from restored outputs and the original coins,
without regenerating the damaged dataset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restoration.pipeline import (CoinRestorationPipeline, IMAGE_SUFFIXES,
                                   to_square, segment_faces)


# Maps each coin_id (filename stem) to its scan in the originals folder.
def build_stem_map(coins_dir):
    stem_map = {}
    for path in sorted(Path(coins_dir).iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            stem_map.setdefault(path.stem, path)
    return stem_map


# Segments one original scan into its faces exactly as prepare_data does (keep
# the two biggest, order left to right) and returns, per face index, the clean
# target and its coin mask as float arrays at the eval resolution.
def clean_and_mask(coin_path, res):
    rgb = np.array(Image.open(coin_path).convert("RGB"))
    faces = segment_faces(rgb)
    if len(faces) > 2:
        faces = sorted(faces, key=lambda f: f.coin_mask.sum(), reverse=True)[:2]
        faces = sorted(faces, key=lambda f: f.bbox[0])

    out = {}
    for face_idx, face in enumerate(faces):
        clean = to_square(face.image, res, fill=1.0, resample=Image.LANCZOS)
        mask = to_square(face.coin_mask, res, fill=0.0, resample=Image.BILINEAR)
        out[face_idx] = (clean.astype(np.float32) / 255.0,
                         (mask > 127).astype(np.float32))
    return out


# Recovers the damaged model input from the left panel of a triptych
# (damaged | restored | clean), as a float array at the eval resolution.
def damaged_from_triptych(triptych_path, res):
    tri = Image.open(triptych_path).convert("RGB")
    third = tri.size[0] // 3
    damaged = tri.crop((0, 0, third, tri.size[1]))
    return np.asarray(damaged.resize((res, res), Image.LANCZOS),
                      dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild results_per_image.csv from restored outputs and "
                    "original coins, without regenerating the damaged dataset.")
    parser.add_argument("--results", required=True,
                        help="infer/eval output folder with "
                             "results_metadata.csv and the restored images")
    parser.add_argument("--coins", required=True,
                        help="folder of original (undamaged) scans, named by "
                             "coin_id")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--ssim", action="store_true",
                        help="also compute SSIM (slower; not needed to rank)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the damaged-vs-clean baseline columns "
                             "(psnr_damaged/lpips_damaged) recovered from each "
                             "triptych's damaged panel")
    parser.add_argument("--baseline-only", action="store_true",
                        help="only add the damaged baseline columns to an "
                             "existing results_per_image.csv, reusing its "
                             "restored scores instead of recomputing them")
    cfg = parser.parse_args()
    if cfg.baseline_only and cfg.no_baseline:
        sys.exit("--baseline-only with --no-baseline does nothing")

    results = Path(cfg.results)
    res = cfg.resolution
    pipe = CoinRestorationPipeline(
        argparse.Namespace(segment=False, resolution=res))

    # In baseline-only mode the restored scores are reused from the existing
    # per-image table; otherwise every restoration is scored from the metadata.
    if cfg.baseline_only:
        source = results / "results_per_image.csv"
        if not source.exists():
            sys.exit(f"--baseline-only needs an existing {source}")
        df = pd.read_csv(source, dtype={"coin_id": str})
        print(f"{len(df)} rows; adding the damaged baseline only")
    else:
        df = pd.read_csv(results / "results_metadata.csv", dtype={"coin_id": str})
        print(f"{len(df)} restored images to score against segmented originals")

    lpips_scorer = None
    if not cfg.no_lpips:
        try:
            lpips_scorer = pipe._lpips_scorer(cfg.device)
        except ImportError:
            print("lpips not installed (pip install lpips); skipping it. "
                  "rank the leakage step with --rank-metric psnr instead")
    ssim_fn = None
    if cfg.ssim:
        from skimage.metrics import structural_similarity
        ssim_fn = structural_similarity

    def score(clean01, restored01, mask01):
        rec = {"psnr": pipe._masked_psnr(clean01, restored01, mask01)}
        if ssim_fn is not None:
            rec["ssim"] = pipe._masked_ssim(clean01, restored01, mask01, ssim_fn)
        if lpips_scorer is not None:
            rec["lpips"] = lpips_scorer(clean01, restored01, mask01)
        return rec

    # The damaged input the model started from is the left panel of each coin's
    # triptych, so the damaged-vs-clean baseline can be recovered without the
    # damaged dataset. It is identical across a base's samples, so score once
    # per triptych and reuse.
    baseline_cache = {}

    def baseline_for(triptych_rel, clean01, mask01):
        if triptych_rel not in baseline_cache:
            rec = {"psnr_damaged": np.nan}
            if lpips_scorer is not None:
                rec["lpips_damaged"] = np.nan
            tri = results / str(triptych_rel)
            if tri.exists():
                damaged01 = damaged_from_triptych(tri, res)
                rec["psnr_damaged"] = pipe._masked_psnr(clean01, damaged01, mask01)
                if lpips_scorer is not None:
                    rec["lpips_damaged"] = lpips_scorer(clean01, damaged01, mask01)
            baseline_cache[triptych_rel] = rec
        return baseline_cache[triptych_rel]

    metric_keys = []
    if not cfg.baseline_only:
        metric_keys += (["psnr"] + (["ssim"] if ssim_fn else [])
                        + (["lpips"] if lpips_scorer else []))
    if not cfg.no_baseline:
        metric_keys += (["psnr_damaged"]
                        + (["lpips_damaged"] if lpips_scorer else []))
    empty = {k: np.nan for k in metric_keys}

    stem_map = build_stem_map(cfg.coins)
    face_cache = {}
    missing = set()

    records = []
    desc = "baseline" if cfg.baseline_only else "scoring"
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="img"):
        coin_id = row["coin_id"]
        if coin_id not in face_cache:
            path = stem_map.get(coin_id)
            face_cache[coin_id] = clean_and_mask(path, res) if path else {}
            if path is None:
                missing.add(coin_id)

        faces = face_cache[coin_id]
        face_idx = int(row["face"])
        if face_idx not in faces:
            records.append(dict(empty))
            continue
        clean01, mask01 = faces[face_idx]
        rec = {}
        if not cfg.baseline_only:
            restored01 = pipe._load01(results / row["restored_path"], res)
            rec.update(score(clean01, restored01, mask01))
        if not cfg.no_baseline:
            rec.update(baseline_for(row["triptych_path"], clean01, mask01))
        records.append(rec)

    new_cols = pd.DataFrame(records)
    if cfg.baseline_only:
        # add/overwrite only the baseline columns, keeping the existing scores
        for column in new_cols.columns:
            df[column] = new_cols[column].values
        full = df
    else:
        full = pd.concat([df.reset_index(drop=True), new_cols], axis=1)
        full["has_chips"] = full["has_chips"].astype(str).str.lower() == "true"
        full["big_clip"] = full["big_clip"].astype(str).str.lower() == "true"
        full["chip_bin"] = full.apply(pipe._chip_bin, axis=1)

    out = results / "results_per_image.csv"
    full.to_csv(out, index=False)
    print(f"-> {out}  ({len(full)} rows)")
    if missing:
        print(f"WARNING: {len(missing)} coin_ids not found in {cfg.coins} "
              f"(e.g. {sorted(missing)[:5]}); their rows are NaN")


if __name__ == "__main__":
    main()
