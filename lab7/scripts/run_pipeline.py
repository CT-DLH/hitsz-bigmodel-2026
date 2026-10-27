"""一键流水线：Blender 导出控制图 -> stable-diffusion.cpp 纯 CPU 生成写实图。

用法示例：
    python scripts/run_pipeline.py --name test512 --res 512 --steps 4 --cfg 1.2
    python scripts/run_pipeline.py --name test768 --res 768 --steps 4 --cfg 1.2
    python scripts/run_pipeline.py --name c05 --control-image output/demo_depth_0001.png --control-strength 0.5
    python scripts/run_pipeline.py --name harmon --init-image output/headfig_face.png --strength 0.35 --control-strength 0.7 --steps 6 --cfg 1.3

每次运行都会把参数、耗时与日志落到 results/<name>.json / results/log_<name>.txt。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_PROMPT = ("photorealistic render, studio soft lighting, PBR material, "
              "preserve original 3d geometry, sharp")
NEG_PROMPT = "deformed geometry, perspective changed"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_sd_cli():
    cands = [
        os.path.join(ROOT, "vendor", "sdcpp", "sd-cli.exe"),
        os.path.join(ROOT, "vendor", "sd-cli.exe"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit("sd-cli.exe not found in vendor/")


def find_blender():
    env = os.environ.get("BLENDER_EXE")
    if env and os.path.exists(env):
        return env
    import glob
    for pat in ("_setup/blender-*/blender.exe", "vendor/blender*/blender.exe"):
        hit = sorted(glob.glob(os.path.join(ROOT, pat)))
        if hit:
            return hit[-1]
    raise SystemExit("blender.exe not found; set BLENDER_EXE")


def run(cmd, log_path, cwd=ROOT):
    log("RUN " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            f.write(line)
            f.flush()
            if any(k in line for k in ("step ", "completed", "error", "Error", "failed", "save")):
                print("   " + line.rstrip()[:160], flush=True)
        p.wait()
    wall = time.time() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed rc={p.returncode}, see {log_path}")
    return wall


def parse_timings(log_path):
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    out = {}
    m = re.search(r"sampling completed,?\s*taking\s*([\d.]+)s", txt)
    if m:
        out["sample_sec"] = float(m.group(1))
    m = re.search(r"(?:decoding|decode)[^\n]{0,40}completed,?\s*taking\s*([\d.]+)s", txt)
    if m:
        out["vae_sec"] = float(m.group(1))
    m = re.search(r"generate_image completed in ([\d.]+)s", txt)
    if m:
        out["total_reported"] = float(m.group(1))
    m = re.search(r"load models from[^\n]*?taking\s*([\d.]+)s", txt)
    if m:
        out["load_sec"] = float(m.group(1))
    # 兜底：新版日志可能只打印 "step 4/4 ... 12.3s/it"
    if "sample_sec" not in out:
        its = [float(x) for x in re.findall(r"([\d.]+)s/it", txt)]
        if its:
            out["sec_per_it"] = sum(its) / len(its)
    return out


def export_maps(name, res, samples):
    blender = find_blender()
    cmd = [blender, "-b", os.path.join(ROOT, "assets", "demo_white_model.blend"),
           "-P", os.path.join(ROOT, "blender", "export_control_maps.py"), "--",
           "--outdir", os.path.join(ROOT, "output"), "--name", name,
           "--res", str(res), "--samples", str(samples)]
    log_path = os.path.join(ROOT, "results", f"log_export_{name}.txt")
    return run(cmd, log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test512")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--control-strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--sampler", default="euler_a")
    ap.add_argument("--control-image", default=None, help="跳过 Blender，直接用已有深度图")
    ap.add_argument("--init-image", default=None, help="img2img 初始图")
    ap.add_argument("--strength", type=float, default=None, help="img2img 去噪强度")
    ap.add_argument("--prompt", default=POS_PROMPT)
    ap.add_argument("--negative", default=NEG_PROMPT)
    ap.add_argument("--export-samples", type=int, default=64)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

    export_sec = None
    if args.control_image is None:
        if not args.skip_export:
            log(f"export control maps for {args.name} @ {args.res}")
            export_sec = export_maps(args.name, args.res, args.export_samples)
            log(f"export done in {export_sec:.1f}s")
        args.control_image = os.path.join(ROOT, "output", f"{args.name}_depth_0001.png")
    control_image = os.path.abspath(args.control_image)
    if not os.path.exists(control_image):
        raise SystemExit("control image missing: " + control_image)

    sd = find_sd_cli()
    out_png = os.path.join(ROOT, "output", f"{args.name}_photoreal.png")
    cmd = [sd,
           "-m", os.path.join(ROOT, "models", "sd_turbo-q4_0.gguf"),
           "--control-net", os.path.join(ROOT, "models", "control_depth_sd21-fp16.safetensors"),
           "--control-image", control_image,
           "--control-strength", str(args.control_strength),
           "-W", str(args.res), "-H", str(args.res),
           "--steps", str(args.steps), "--cfg-scale", str(args.cfg),
           "--sampling-method", args.sampler,
           "-s", str(args.seed), "-t", str(args.threads),
           "-p", args.prompt, "-n", args.negative,
           "-o", out_png, "--color"]
    if args.init_image:
        cmd += ["-i", os.path.abspath(args.init_image), "--strength", str(args.strength or 0.5)]

    log_path = os.path.join(ROOT, "results", f"log_{args.name}.txt")
    wall = run(cmd, log_path)
    tim = parse_timings(log_path)
    rec = {
        "name": args.name, "res": args.res, "steps": args.steps, "cfg": args.cfg,
        "control_strength": args.control_strength, "seed": args.seed, "sampler": args.sampler,
        "threads": args.threads, "control_image": os.path.relpath(control_image, ROOT),
        "init_image": os.path.relpath(args.init_image, ROOT) if args.init_image else None,
        "strength": args.strength, "prompt": args.prompt, "negative": args.negative,
        "wall_sec": round(wall, 2), "export_sec": round(export_sec, 2) if export_sec else None,
        "output": os.path.relpath(out_png, ROOT), "output_exists": os.path.exists(out_png),
        "output_bytes": os.path.getsize(out_png) if os.path.exists(out_png) else 0,
        "log": os.path.relpath(log_path, ROOT),
    }
    rec.update({k: round(v, 2) for k, v in tim.items()})
    with open(os.path.join(ROOT, "results", f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
