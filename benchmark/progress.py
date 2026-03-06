"""Global progress tracker with nested step / substep display.

Uses ``tqdm`` to render a top-level progress bar tracking benchmark steps
and prints timing summaries at the end.
"""

from __future__ import annotations

import time
from typing import Dict, List

from tqdm import tqdm

from benchmark.config import BenchmarkConfig


class BenchmarkProgress:
    """Track benchmark progress across multiple pipeline steps.

    Parameters
    ----------
    config:
        Immutable benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._step_timings: Dict[str, float] = {}
        self._step_start: float = 0.0

        self._steps = self._build_steps(config)
        self._total = len(self._steps)
        self._current_idx = 0

        self._global_bar = tqdm(
            total=self._total,
            desc=f"Benchmark {config.dataset}/{config.embedding}",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} steps "
                "[{elapsed}<{remaining}, {postfix}]"
            ),
            position=0,
            leave=True,
            colour="green",
        )
        if self._steps:
            self._global_bar.set_postfix_str(self._steps[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_step(self, step_name: str) -> None:
        """Mark the start of a step (prints banner + updates bar)."""
        self._step_start = time.time()
        self._global_bar.set_postfix_str(step_name)
        tqdm.write("")
        tqdm.write("=" * 70)
        tqdm.write(f"[{self._current_idx}/{self._total}] {step_name}")
        tqdm.write("=" * 70)

    def end_step(self, step_name: str) -> None:
        """Mark step completion and advance the global bar."""
        elapsed = time.time() - self._step_start
        self._step_timings[step_name] = elapsed
        self._current_idx += 1
        self._global_bar.update(1)
        mins, secs = divmod(int(elapsed), 60)
        tqdm.write(f"  -> {step_name} done in {mins}m{secs:02d}s")

    def close(self, total_elapsed: float) -> None:
        """Print final timing summary and close bars."""
        self._global_bar.set_postfix_str("COMPLETE")
        self._global_bar.close()
        tqdm.write("")
        tqdm.write("=" * 70)
        tqdm.write("BENCHMARK TIMING SUMMARY")
        tqdm.write("=" * 70)
        for name, secs in self._step_timings.items():
            self._print_timing_line(name, secs)
        tqdm.write("-" * 70)
        self._print_timing_line("TOTAL", total_elapsed)
        tqdm.write("=" * 70)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_steps(config: BenchmarkConfig) -> List[str]:
        """Derive ordered step names from configuration."""
        steps = ["Step 0: Scaffold Splits"]
        if 2 in config.levels:
            steps.append("Step 0b: Ligand Vectors")
        if config.finetune:
            steps.append("Step FT: ESM-2 + MolFormer Fine-tuning")
        if 1 in config.levels:
            steps.append("Step 1: Level 1 (FP+KNN/MLP)")
        if 2 in config.levels:
            steps.append("Step 2: Level 2 (Emb+KNN/MLP)")
        if 3 in config.levels:
            steps.append("Step 3: Level 3 (Mat+MeanPool+KNN/MLP)")
        if 4 in config.levels:
            steps.append("Step 4: Level 4 (CrossAtt+KNN/MLP)")
        steps.append("Report + Visualizations")
        return steps

    @staticmethod
    def _print_timing_line(label: str, seconds: float) -> None:
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            tqdm.write(f"  {label:<42s}  {hours}h{minutes:02d}m{secs:02d}s")
        else:
            tqdm.write(f"  {label:<42s}  {minutes}m{secs:02d}s")
