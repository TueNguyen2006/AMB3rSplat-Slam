"""Continue a fixed SLAM map to absolute global-step milestones."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .gaussian_mapper import IncrementalGaussianMapper
from .state import SharedReconstructionState

ROOT = Path(__file__).resolve().parents[1]


def fixed_digest(keyframes):
    digest = hashlib.sha256()
    for frame, item in sorted(keyframes.items()):
        digest.update(str(frame).encode())
        for key, value in sorted(item.items()):
            if isinstance(value, torch.Tensor):
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MemoryMonitor:
    def __init__(self):
        self.peak_mib = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()

    def poll(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                self.peak_mib = max(self.peak_mib, int(result.stdout.splitlines()[0]))
            except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(1)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/amb3r_gs_slam/full_quality")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/amb3r_gs_slam/refinement")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--milestones", type=int, nargs="+", default=[3000, 10000, 30000])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.resume or args.source / "checkpoints/final.pt"
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Refinement must use a separate output directory")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(source, map_location="cpu", weights_only=False)
    shared = SharedReconstructionState(args.output)
    mapper = IncrementalGaussianMapper(ROOT / "thirdparty/neural-harmonic-textures/gsplat",
                                       shared, max_gaussians=1000000, sh_degree=3)
    mapper.load_state_dict(payload["mapper"])
    initial_global = mapper.stats["global_steps"]
    assert mapper.sh_degree == 3
    original_digest = fixed_digest(mapper.keyframes)
    if "refinement" in payload:
        assert payload["refinement"]["fixed_digest"] == original_digest
        torch.set_rng_state(payload["rng"]["torch"])
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
    metadata = {k: v for k, v in payload.items() if k not in ("mapper", "rng", "refinement")}
    previous_seconds = payload.get("refinement", {}).get("training_seconds", 0.0)
    del payload
    frames = sorted(mapper.keyframes)
    monitor = MemoryMonitor()
    torch.cuda.reset_peak_memory_stats()
    losses_path = args.output / "loss.jsonl"
    if losses_path.exists() and args.resume is None:
        raise FileExistsError("Existing refinement: specify --resume; refusing to overwrite")
    started = time.perf_counter()
    train_seconds = previous_seconds

    def save(path, step):
        assert fixed_digest(mapper.keyframes) == original_digest, "Fixed SLAM data changed"
        data = dict(metadata)
        data.update(mapper=mapper.state_dict(), refinement={
            "global_steps": step, "training_seconds": train_seconds,
            "fixed_digest": original_digest, "source": str(source),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "nvidia_smi_peak_mib": monitor.peak_mib,
        }, rng={"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        tmp.replace(path)

    try:
        with losses_path.open("a", buffering=1) as log:
            for global_step in range(initial_global, max(args.milestones)):
                window = [frames[(global_step * 4 + j) % len(frames)]
                          for j in range(min(8, len(frames)))]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                result = mapper.mapping_step(window)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                train_seconds += elapsed
                if not np.isfinite(result["loss"]):
                    raise FloatingPointError(f"Nonfinite loss at {global_step + 1}")
                mapper.stats["global_steps"] = global_step + 1
                row = dict(result, global_step=global_step + 1, seconds=elapsed,
                           training_seconds=train_seconds)
                log.write(json.dumps(row) + "\n")
                if (global_step + 1) % 100 == 0:
                    print(json.dumps(row), flush=True)
                if (global_step + 1) % 1000 == 0:
                    save(args.output / "checkpoints/latest.pt", global_step + 1)
                if global_step + 1 in args.milestones:
                    out = args.output / f"step_{global_step + 1:06d}"
                    save(out / "checkpoints/final.pt", global_step + 1)
                    shutil.copy2(args.source / "trajectory.npz", out / "trajectory.npz")
                    report = dict(row, initial_global_steps=initial_global,
                                  fixed_digest=original_digest,
                                  peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                                  peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                                  nvidia_smi_peak_mib=monitor.peak_mib,
                                  wall_seconds=time.perf_counter() - started)
                    (out / "refinement_report.json").write_text(json.dumps(report, indent=2))
                    print(f"MILESTONE {global_step + 1}: {out}", flush=True)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
