"""send_adm_osc.py — Step 6 of the CRE pipeline.

Reads a pre-calculated spatial timeline CSV and dispatches ADM-OSC messages
to Nuendo in real time, either:
  - immediately (default): starts as soon as the script runs, useful for
    testing and rehearsal.
  - synchronized (--wait-for-nuendo): the script listens on a UDP port for
    a /transport/play message from Nuendo, then starts the scheduler.

✅ Formato OSC confermato per Nuendo (ADM-OSC EBU):
    Address : /adm/obj/{n}/xyz
    Args    : [x (float -1..+1), y (float -1..+1), z (float 0..1)]
    Port    : 8000
    Note    : x=-1.0 = sinistra, x=+1.0 = destra, y=0.0 = centro asse Y
              La pipeline converte automaticamente x/y da [0,1] a [-1,+1]

Usage examples:
    # Immediate playback (for testing)
    python scripts/send_adm_osc.py --timeline outputs/spatial_timeline.csv

    # Dry-run (print messages without sending)
    python scripts/send_adm_osc.py --timeline outputs/spatial_timeline.csv --dry-run

    # Synchronized with Nuendo (press PLAY in Nuendo first)
    python scripts/send_adm_osc.py \\
        --timeline outputs/spatial_timeline.csv \\
        --wait-for-nuendo \\
        --listen-port 9001

    # Custom Nuendo address
    python scripts/send_adm_osc.py \\
        --timeline outputs/spatial_timeline.csv \\
        --host 192.168.1.10 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT / "src"))

from cre.rule_engine import OscScheduler
from pythonosc.udp_client import SimpleUDPClient

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_DEFAULT_OSC_HOST = "127.0.0.1"
_DEFAULT_OSC_PORT = 8000
_DEFAULT_LISTEN_PORT = 9001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRE Step 6 — Send ADM-OSC messages from a pre-calculated spatial timeline.",
    )
    parser.add_argument(
        "--timeline",
        required=True,
        help="Path to the spatial timeline CSV (output of generate_spatial_timeline.py).",
    )
    parser.add_argument(
        "--host",
        default=_DEFAULT_OSC_HOST,
        help=f"Nuendo OSC host (default: {_DEFAULT_OSC_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_OSC_PORT,
        help=f"Nuendo OSC port (default: {_DEFAULT_OSC_PORT}).",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=_DEFAULT_LISTEN_PORT,
        help=f"UDP port to listen for /transport/play from Nuendo (default: {_DEFAULT_LISTEN_PORT}).",
    )
    parser.add_argument(
        "--wait-for-nuendo",
        action="store_true",
        help="Block until a /transport/play OSC message is received from Nuendo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print OSC messages to stdout without opening any UDP socket.",
    )
    parser.add_argument(
        "--no-cartesian",
        action="store_true",
        help="Only send spread values, skip /cartesian position messages.",
    )
    return parser.parse_args()


def send_messages(
    timeline_path: Path,
    host: str = _DEFAULT_OSC_HOST,
    port: int = _DEFAULT_OSC_PORT,
    listen_port: int = _DEFAULT_LISTEN_PORT,
    wait_for_nuendo: bool = False,
    dry_run: bool = False,
) -> None:
    """Dispatch OSC messages from a spatial timeline CSV.

    Args:
        timeline_path:    Path to the CSV produced by generate_spatial_timeline.py.
        host:             Nuendo OSC host.
        port:             Nuendo OSC receive port.
        listen_port:      Local UDP port to listen for /transport/play.
        wait_for_nuendo:  If True, block until /transport/play is received.
        dry_run:          If True, log messages without sending.
    """
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline CSV not found: {timeline_path}")

    LOGGER.info("Loading timeline: %s", timeline_path)
    timeline_df = pd.read_csv(timeline_path)
    LOGGER.info("Loaded %d rows from timeline.", len(timeline_df))

    # Validate required columns
    required = {"timestamp_s", "object_index", "spread", "x", "y", "z"}
    missing = required - set(timeline_df.columns)
    if missing:
        raise ValueError(
            f"Timeline CSV is missing required columns: {missing}. "
            "Re-generate the timeline with generate_spatial_timeline.py."
        )

    # Create OSC client (or None in dry-run mode)
    osc_client: SimpleUDPClient | None = None
    if not dry_run:
        osc_client = SimpleUDPClient(host, port)
        LOGGER.info("OSC client → %s:%d", host, port)
    else:
        LOGGER.info("DRY-RUN mode — no UDP socket opened.")

    scheduler = OscScheduler(
        timeline=timeline_df,
        client=osc_client,
        dry_run=dry_run,
        listen_host="0.0.0.0",
        listen_port=listen_port,
    )

    if wait_for_nuendo:
        LOGGER.info(
            "Waiting for /transport/play from Nuendo on UDP port %d…  "
            "(Press PLAY in Nuendo)",
            listen_port,
        )
        scheduler.start(t0_wall=None)   # blocks until START received
    else:
        LOGGER.info("Starting immediately (no Nuendo sync)…")
        scheduler.start(t0_wall=time.perf_counter())

    scheduler.join()
    LOGGER.info("✅  All OSC messages dispatched.")


def main() -> None:
    args = parse_args()
    timeline_path = Path(args.timeline).expanduser().resolve()
    send_messages(
        timeline_path=timeline_path,
        host=args.host,
        port=args.port,
        listen_port=args.listen_port,
        wait_for_nuendo=args.wait_for_nuendo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
