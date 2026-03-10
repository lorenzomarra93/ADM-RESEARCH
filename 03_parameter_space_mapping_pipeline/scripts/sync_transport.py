"""sync_transport.py — Transport synchronisation helper for the CRE pipeline.

Bridges between a DAW (Nuendo) and the Python OSC scheduler by:
  1. Listening for ADM-OSC transport messages from Nuendo:
       /transport/play   → start scheduler (passes t=0 reference)
       /transport/stop   → stop scheduler
       /transport/locate → jump to a specific timecode (future use)
  2. Optionally relaying MTC (MIDI Time Code) from a MIDI interface to
     derive an accurate wall-clock offset (future use).

This module can be used standalone (run as __main__) or imported by
rule_engine.py / send_adm_osc.py.

Usage:
    python scripts/sync_transport.py --listen-port 9001 [--verbose]

The script will print a line each time a transport message arrives.
It does NOT send any OSC messages itself — pair it with send_adm_osc.py
or run rule_engine.py --realtime for the complete pipeline.

ADM-OSC transport messages (EBU ADM-OSC spec):
    /transport/play      → begin playback from current position
    /transport/stop      → halt playback
    /transport/locate    → float: locate to timecode (seconds)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT / "src"))

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_DEFAULT_LISTEN_HOST = "0.0.0.0"
_DEFAULT_LISTEN_PORT = 9001


# ──────────────────────────────────────────────
# TRANSPORT STATE
# ──────────────────────────────────────────────

class TransportState:
    """Shared mutable transport state updated by OSC callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_playing: bool = False
        self.t0_wall: Optional[float] = None        # perf_counter at last PLAY
        self.locate_seconds: float = 0.0            # last /transport/locate value

    # ── callbacks ──────────────────────────────────────────────────────────

    def on_play(self, address: str, *args) -> None:  # noqa: ANN002
        with self._lock:
            self.is_playing = True
            self.t0_wall = time.perf_counter()
        LOGGER.info("▶  PLAY received (%s) — t0_wall=%.4fs", address, self.t0_wall)

    def on_stop(self, address: str, *args) -> None:  # noqa: ANN002
        with self._lock:
            self.is_playing = False
        LOGGER.info("■  STOP received (%s)", address)

    def on_locate(self, address: str, *args) -> None:  # noqa: ANN002
        if args:
            with self._lock:
                self.locate_seconds = float(args[0])
        LOGGER.info("⏩ LOCATE received (%s) → %.3fs", address, self.locate_seconds)

    # ── helpers ────────────────────────────────────────────────────────────

    def elapsed_seconds(self) -> float:
        """Current playback position in seconds, or 0 if stopped."""
        with self._lock:
            if not self.is_playing or self.t0_wall is None:
                return self.locate_seconds
            return self.locate_seconds + (time.perf_counter() - self.t0_wall)

    def wait_for_play(self, timeout: float = 300.0) -> bool:
        """Block until a PLAY message arrives or timeout expires.

        Returns True if PLAY was received, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.is_playing:
                    return True
            time.sleep(0.01)
        return False


# ──────────────────────────────────────────────
# OSC LISTENER
# ──────────────────────────────────────────────

class TransportListener:
    """OSC server that updates a TransportState from incoming messages.

    Can be started in a background thread so other code keeps running.
    """

    def __init__(
        self,
        host: str = _DEFAULT_LISTEN_HOST,
        port: int = _DEFAULT_LISTEN_PORT,
        on_play_callback: Optional[Callable] = None,
        on_stop_callback: Optional[Callable] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.state = TransportState()
        self._on_play_callback = on_play_callback
        self._on_stop_callback = on_stop_callback
        self._server: Optional[BlockingOSCUDPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _build_dispatcher(self) -> Dispatcher:
        d = Dispatcher()

        def _play(address, *args):
            self.state.on_play(address, *args)
            if self._on_play_callback:
                self._on_play_callback(address, *args)

        def _stop(address, *args):
            self.state.on_stop(address, *args)
            if self._on_stop_callback:
                self._on_stop_callback(address, *args)

        d.map("/transport/play", _play)
        d.map("/transport/stop", _stop)
        d.map("/transport/locate", self.state.on_locate)
        d.set_default_handler(
            lambda addr, *a: LOGGER.debug("Unhandled OSC: %s %s", addr, a)
        )
        return d

    def start_background(self) -> None:
        """Start the OSC listener in a daemon thread."""
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        LOGGER.info(
            "Transport listener started on %s:%d (background thread)",
            self.host,
            self.port,
        )

    def _serve_forever(self) -> None:
        try:
            dispatcher = self._build_dispatcher()
            self._server = BlockingOSCUDPServer((self.host, self.port), dispatcher)
            LOGGER.info("Listening on %s:%d…", self.host, self.port)
            self._server.serve_forever()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Transport listener error: %s", exc)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            LOGGER.info("Transport listener stopped.")


# ──────────────────────────────────────────────
# sync_to_daw() — public API
# ──────────────────────────────────────────────

def sync_to_daw(
    listen_host: str = _DEFAULT_LISTEN_HOST,
    listen_port: int = _DEFAULT_LISTEN_PORT,
    timeout: float = 300.0,
) -> Optional[float]:
    """Block until a /transport/play message is received from the DAW.

    Returns the wall-clock perf_counter timestamp of the PLAY event,
    or None if timed out.

    This is the function called by rule_engine.py and send_adm_osc.py
    when --wait-for-nuendo / --realtime flags are used.

    Args:
        listen_host:  Network interface to bind (default: all interfaces).
        listen_port:  UDP port to receive OSC messages.
        timeout:      Maximum seconds to wait (default: 5 minutes).

    Returns:
        perf_counter() value at the moment PLAY was received, or None.
    """
    listener = TransportListener(host=listen_host, port=listen_port)
    listener.start_background()
    LOGGER.info(
        "Waiting for /transport/play on %s:%d  (timeout: %.0fs)…",
        listen_host,
        listen_port,
        timeout,
    )
    received = listener.state.wait_for_play(timeout=timeout)
    if received:
        t0 = listener.state.t0_wall
        LOGGER.info("Sync established — t0_wall = %.4f", t0)
        return t0
    else:
        LOGGER.warning("Timeout waiting for /transport/play — no sync established.")
        listener.stop()
        return None


# ──────────────────────────────────────────────
# CLI — standalone monitor
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CRE transport monitor — listens for /transport/play|stop|locate "
            "messages from Nuendo and reports them to the console."
        )
    )
    parser.add_argument(
        "--listen-host",
        default=_DEFAULT_LISTEN_HOST,
        help=f"Interface to bind (default: {_DEFAULT_LISTEN_HOST}).",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=_DEFAULT_LISTEN_PORT,
        help=f"UDP port to listen on (default: {_DEFAULT_LISTEN_PORT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    listener = TransportListener(host=args.listen_host, port=args.listen_port)

    print(
        f"\n🎧  CRE Transport Monitor\n"
        f"   Listening on {args.listen_host}:{args.listen_port}\n"
        f"   Configure Nuendo to send OSC to this machine on port {args.listen_port}.\n"
        f"   Press PLAY / STOP in Nuendo — messages will appear here.\n"
        f"   Ctrl+C to quit.\n"
    )

    try:
        # Run the server in the foreground (blocking)
        dispatcher = listener._build_dispatcher()
        server = BlockingOSCUDPServer((args.listen_host, args.listen_port), dispatcher)
        LOGGER.info("Server ready on %s:%d", args.listen_host, args.listen_port)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
