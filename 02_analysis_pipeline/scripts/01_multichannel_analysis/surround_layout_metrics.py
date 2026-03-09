"""Placeholder for multichannel (bed-based) analysis utilities."""

from pathlib import Path

def analyze_surround_layout(audio_path: Path, layout: str = "7.1.4") -> None:
    """Inspect a multichannel file and compute layout-aware metrics."""
    # TODO: integrate ITU channel order parsing and loudness calculations
    raise NotImplementedError("Multichannel analysis not implemented yet")

if __name__ == "__main__":
    print("surround_layout_metrics.py placeholder - supply multichannel beds (e.g., 7.1)")
