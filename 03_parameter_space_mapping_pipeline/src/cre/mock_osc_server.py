"""Mock OSC server for validating ADM-OSC packets without Nuendo/SPAT."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from pythonosc import dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def handle_message(address: str, *values: float) -> None:
    """Print every inbound OSC packet with a timestamp."""

    timestamp = datetime.now().isoformat(timespec="milliseconds")
    LOGGER.info("%s | %s | %s", timestamp, address, ", ".join(f"{v:.3f}" for v in values))


def run_server(ip: str, port: int) -> None:
    """Start a blocking OSC server that logs any ADM-OSC address."""

    disp = dispatcher.Dispatcher()
    disp.set_default_handler(lambda addr, *args: handle_message(addr, *args))
    server = ThreadingOSCUDPServer((ip, port), disp)
    LOGGER.info("Mock OSC server listening on %s:%d", ip, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down mock server.")


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""

    parser = argparse.ArgumentParser(description="Mock ADM-OSC receiver.")
    parser.add_argument("--ip", default="127.0.0.1", help="IP address to bind to.")
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_server(cli_args.ip, cli_args.port)
