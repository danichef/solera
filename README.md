# Solera

Generative pattern restoration for damaged relief surfaces, applied to ancient
silver coins. This is my MSc dissertation project.

The idea is simple to state: take an object whose surface pattern has been
worn down over time and teach a model to bring the original back. Getting
there is the interesting part, because no dataset of before/after pairs
exists. Nobody photographed these coins two thousand years ago.

Solera solves that with synthetic supervision. A classical image pipeline
simulates the physics of wear on clean coin scans: it estimates which parts
of the relief are raised, grinds them down with a simulated sanding plane,
relights the result so it reads as handled metal, and bites chips out of the
rim. Every clean scan becomes several damaged versions of itself, each with a
known ground truth. An SDXL InstructPix2Pix model is then fine-tuned on those
pairs to learn the inverse mapping, from damaged back to original.

The name comes from the solera system used to age sherry in Andalusia, where
old stock is continuously refreshed with young wine. Old material, brought
back to life. That felt right for a model that restores Roman-era silver.

## How it works

1. **Damage generation.** Coin scans are segmented into faces and each face is
   damaged at four wear grades. Chips are sampled independently of wear,
   since a well-preserved coin can still have a nicked rim.
2. **Training.** SDXL is fine-tuned as an instruction-based image editor with
   a masked loss that only scores pixels inside the original coin silhouette.
   If the model also repairs damage the real coin already had, it is not
   punished for it.
3. **Evaluation.** Restored test images are scored with masked PSNR, SSIM and
   LPIPS, broken down by the exact damage state each image was restored from.

The whole thing runs from one command. It validates itself on a tiny smoke
run first, then generates data, trains, tunes its own inference settings on
the validation split, and evaluates. Every stage skips itself if its output
already exists, so an interrupted run continues where it stopped.

```bash
python restoration/pipeline.py run --src <raw scans> --data <dataset dir> \
  --output-dir <model dir> --results <results dir>
```

Individual stages are available as subcommands (`prepare`, `train`, `infer`,
`eval`, `restore`). The `restore` subcommand applies a trained model to any
new image.

## Layout

```
damage/                  the wear and chip simulator (numpy + scipy, no ML)
restoration/pipeline.py  the full pipeline in one class
requirements.txt         dependencies
```

Training was done on a single rented A100. The damage simulator runs on any
CPU. Model weights are not in this repo due to their size.

## Status

Work in progress as part of my MSc dissertation. The damage simulator and
pipeline are stable; training experiments are ongoing.
