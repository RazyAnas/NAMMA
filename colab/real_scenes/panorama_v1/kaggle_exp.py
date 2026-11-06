"""Launch a panorama experiment on Kaggle (2x T4) and collect small outputs.

The kernel is one generated script: a base64 tarball of this folder's .py
files plus the experiment config. Nothing is hand-quoted into strings, and
no separate code dataset has to propagate before the kernel starts.

  python kaggle_exp.py push  --tag h01 --config cfg.json
  python kaggle_exp.py wait  --tag h01            # poll until complete/error
  python kaggle_exp.py fetch --tag h01 --dest DIR # small files only
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMALL = r".*(json|log|jpg|png|txt)$"


def kaggle(*args, check=True) -> str:
    exe = Path(sys.executable).with_name("kaggle")
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"kaggle {' '.join(map(str, args))}: {p.stdout[-800:]} {p.stderr[-800:]}")
    return (p.stdout or "") + (p.stderr or "")


def user() -> str:
    return "anasrazy"


def kernel_id(tag: str) -> str:
    return f"{user()}/namma-pano-exp-{tag}"


def build_script(config: dict) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(HERE.glob("*.py")):
            tar.add(path, arcname=path.name)
    blob = base64.b64encode(buf.getvalue()).decode()
    return (
        "import base64, io, json, os, sys, tarfile\n"
        f"BLOB = {blob!r}\n"
        f"CONFIG = {json.dumps(config)!r}\n"
        "code = '/kaggle/temp/namma-code'\n"
        "os.makedirs(code, exist_ok=True)\n"
        "tarfile.open(fileobj=io.BytesIO(base64.b64decode(BLOB))).extractall(code)\n"
        "sys.path.insert(0, code)\n"
        "os.environ['NAMMA_EXP'] = CONFIG\n"
        "import exp_kernel\n"
        "exp_kernel.HERE = __import__('pathlib').Path(code)\n"
        "sys.argv = ['exp_kernel']\n"
        "exp_kernel.main()\n"
    )


def push(tag: str, config: dict) -> str:
    kid = kernel_id(tag)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(build_script(config), encoding="utf-8")
        meta = {
            "id": kid, "title": kid.split("/")[1], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "kernel_sources": config.get("kernel_sources", []),
            "competition_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(kaggle("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4")[-600:])
    return kid


def status(tag: str) -> str:
    return kaggle("kernels", "status", kernel_id(tag), check=False).strip()


def wait(tag: str, poll: int = 120, max_hours: float = 11.5) -> str:
    seen_running = False
    t0 = time.time()
    while time.time() - t0 < max_hours * 3600:
        s = status(tag)
        low = s.lower()
        if "running" in low or "queued" in low:
            seen_running = True
        if seen_running and ("complete" in low or "error" in low or "cancel" in low):
            print(s, flush=True)
            return s
        if not seen_running and time.time() - t0 > 900 and ("complete" in low or "error" in low):
            print(s, flush=True)
            return s
        time.sleep(poll)
    return "timeout"


def fetch(tag: str, dest: Path, pattern: str = SMALL) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel_id(tag), "-p", dest, "-o", "-q",
                 "--file-pattern", pattern, check=False)[-400:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "wait", "fetch", "status", "script"])
    p.add_argument("--tag", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--dest", type=Path)
    p.add_argument("--pattern", default=SMALL)
    a = p.parse_args()
    if a.action == "push":
        print(push(a.tag, json.loads(a.config.read_text())))
    elif a.action == "wait":
        print(wait(a.tag))
    elif a.action == "status":
        print(status(a.tag))
    elif a.action == "fetch":
        fetch(a.tag, a.dest, a.pattern)
    elif a.action == "script":
        sys.stdout.write(build_script(json.loads(a.config.read_text()))[:2000])


if __name__ == "__main__":
    main()
