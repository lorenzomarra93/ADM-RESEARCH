"""Entry point for ADM/BW64 parsing."""

from pathlib import Path

def parse_adm_file(source_path: Path) -> dict:
    """Parse an ADM-enabled BW64 file into a normalized dictionary."""
    # TODO: integrate actual parser (ebu_adm or custom implementation)
    raise NotImplementedError("ADM parsing not implemented yet")

def main() -> None:
    """CLI placeholder."""
    # TODO: wire up argparse and batch processing
    print("parse_adm.py placeholder - configure via configs/pipeline_config.yaml")

if __name__ == "__main__":
    main()
