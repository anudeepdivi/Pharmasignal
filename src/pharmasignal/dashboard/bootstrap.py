"""Generate any missing dashboard artifacts (used on first start, e.g. on a fresh cloud host).

Runs the same CLIs as ``make artifacts`` (the model step uses plain training rather
than MLflow tracking; it produces the same model and metrics). Each step writes into a
temporary sibling directory and is renamed into place only on success, so an
interrupted run never leaves a half-written directory that looks complete.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("shifted", "unchanged")


@dataclass(frozen=True)
class Step:
    name: str
    output: Path  # directory the step creates
    marker: str  # file whose presence means the step already completed
    args: tuple[str, ...]  # module + arguments; "{out}" is replaced by the temp output dir


def plan(data: Path, model: Path, simulated: Path, monitoring: Path) -> list[Step]:
    steps = [
        Step(
            "Generating synthetic dataset",
            data,
            "manifest.json",
            ("pharmasignal.data.build", "--config", str(ROOT / "configs/data.json"), "--output"),
        ),
        Step(
            "Training the model",
            model,
            "metadata.json",
            ("pharmasignal.models.train", "--data", str(data), "--output"),
        ),
    ]
    for scenario in SCENARIOS:
        steps.append(
            Step(
                f"Simulating {scenario} batch",
                simulated / scenario,
                "manifest.json",
                ("pharmasignal.monitoring.simulate", "--scenario", scenario, "--n", "600")
                + ("--output",),
            )
        )
    for scenario in SCENARIOS:
        steps.append(
            Step(
                f"Building {scenario} drift report",
                monitoring / scenario,
                "report.json",
                (
                    "pharmasignal.monitoring.report",
                    "--reference",
                    str(data),
                    "--batch",
                    str(simulated / scenario),
                    "--config",
                    str(ROOT / "configs/monitoring.json"),
                    "--output",
                ),
            )
        )
    return steps


def pending(steps: list[Step]) -> list[Step]:
    return [step for step in steps if not (step.output / step.marker).exists()]


def run_step(step: Step) -> None:
    if step.output.exists():
        raise RuntimeError(
            f"{step.output} exists but is incomplete (no {step.marker}); delete it and retry"
        )
    step.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = step.output.with_name(f".{step.output.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
        ),
    }
    try:
        subprocess.run(
            [sys.executable, "-m", *step.args, str(temporary)],
            check=True,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        temporary.rename(step.output)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"{step.name} failed: {detail[0]}") from None
