"""
Generate images from prompts.

Examples
--------
One prompt:
    python generate.py --prompt "a dog on a sunny beach, long shadow"

Many prompts (one per line), 3 seeds each:
    python generate.py --prompts-file prompts.txt --seeds 0 1 2

Also save U-Net activations (16 self-attention layers, pooled to 16x16, steps 0-4 and 14):
    python generate.py --prompts-file prompts.txt --save-activations --act-pool 16 --act-steps 0 1 2 3 4 14

Output (in --out-dir, default outputs/):
    images/00000_s0.png ...
    activations/00000_s0.pt   (with --save-activations) dict: steps, timesteps, acts{layer: [n_steps,C,H,W]}, latents
    metadata.jsonl            one JSON line per image (prompt, seed, steps, timesteps, ...)
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from capture import ActivationRecorder, sd_self_attention_layers
from sd_generator import DEFAULT_MODEL, SCHEDULERS, SDGenerator


def read_prompts(args) -> list[str]:
    if args.prompt:
        return [args.prompt]
    lines = Path(args.prompts_file).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str)
    src.add_argument("--prompts-file", type=str)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--size", type=int, default=None, help="image side in px (default: model native, 512 for SD v1)")
    p.add_argument("--scheduler", choices=list(SCHEDULERS), default="lms")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--out-dir", type=str, default="outputs")
    # activation capture
    p.add_argument("--save-activations", action="store_true")
    p.add_argument("--act-steps", type=int, nargs="+", default=None, help="step indices to keep (default: all)")
    p.add_argument("--act-pool", type=int, default=None, help="pool maps to NxN (default: native resolution)")
    p.add_argument("--act-layers", type=str, nargs="+", default=None, help="layer names, e.g. mid dec2_l1 (default: all 16)")
    args = p.parse_args()

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"

    prompts = read_prompts(args)
    print(f"Loading {args.model} ...")
    gen = SDGenerator(args.model, scheduler=args.scheduler, device=args.device)
    print(f"device={gen.device} dtype={gen.dtype} | {len(prompts)} prompt(s) x {len(args.seeds)} seed(s)")

    rec = None
    if args.save_activations:
        (out / "activations").mkdir(exist_ok=True)
        layers = sd_self_attention_layers(gen.unet)
        if args.act_layers:
            layers = {n: layers[n] for n in args.act_layers}
        rec = ActivationRecorder(gen, layers=layers, steps=args.act_steps, pool=args.act_pool)
        print(f"capturing {len(layers)} layers: {', '.join(layers)}")

    with meta_path.open("a") as f:
        for pi, prompt in enumerate(prompts):
            for seed in args.seeds:
                t0 = time.time()
                with rec if rec else nullcontext():
                    res = gen.generate(
                        prompt,
                        seed=seed,
                        num_steps=args.steps,
                        guidance_scale=args.guidance,
                        negative_prompt=args.negative_prompt,
                        height=args.size,
                        width=args.size,
                        step_callback=rec.record_latents if rec else None,
                    )
                stem = f"{pi:05d}_s{seed}"
                name = f"{stem}.png"
                res.image.save(out / "images" / name)
                if rec:
                    data = rec.pop()
                    data["timesteps"] = [res.timesteps[s] for s in data["steps"]]
                    torch.save(data, out / "activations" / f"{stem}.pt")
                f.write(json.dumps(dict(
                    file=f"images/{name}",
                    activations=f"activations/{stem}.pt" if rec else None,
                    prompt_index=pi,
                    prompt=prompt,
                    seed=seed,
                    steps=res.num_steps,
                    guidance=res.guidance_scale,
                    timesteps=res.timesteps,
                    **res.meta,
                )) + "\n")
                f.flush()
                print(f"[{pi + 1}/{len(prompts)}] seed={seed} {time.time() - t0:.1f}s -> {name}")


if __name__ == "__main__":
    main()
