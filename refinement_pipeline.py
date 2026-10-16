"""Serial milestone training and diagnostics with recovery checkpoints."""
import argparse
import subprocess
import sys
from pathlib import Path

from .refine_long import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/amb3r_gs_slam/refinement")
    args=parser.parse_args()
    output=args.output.resolve()
    source=ROOT / "outputs/amb3r_gs_slam/full_quality"
    def run(module, *arguments):
        subprocess.run([sys.executable,"-m",module,*map(str,arguments)], cwd=ROOT, check=True)
    for step in (1000,3000,10000,30000):
        target=output / f"step_{step:06d}"
        if step>1000 and not (target / "checkpoints/final.pt").exists():
            resume=output / "checkpoints/latest.pt"
            arguments=["--output",output,"--milestones",step]
            if resume.exists():
                arguments += ["--resume",resume]
            run("amb3r_gs_slam.refine_long",*arguments)
        if not (target / "diagnostics/diagnostics.json").exists():
            run("amb3r_gs_slam.refinement_diagnostics", "--run",source if step==1000 else target,
                "--output",target / "diagnostics")
        if not (target / "fixed_training_loss.json").exists():
            run("amb3r_gs_slam.refinement_loss", "--run",source if step==1000 else target,
                "--output",target / "fixed_training_loss.json")
        if step>1000:
            run("amb3r_gs_slam.refinement_export","--run",target)
        if not (target / "gaussian_geometry.json").exists():
            run("amb3r_gs_slam.refinement_geometry","--run",source if step==1000 else target,
                "--output",target / "gaussian_geometry.json")
        if (output / "loss.jsonl").exists():
            run("amb3r_gs_slam.refinement_report","--root",output)


if __name__=="__main__":
    main()
