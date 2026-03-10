"""
OSC Discovery Tool v2 — ricerca sistematica formato+porta per Nuendo.

Modalità disponibili:
  --scan-ports   : prova TUTTE le porte candidate (5s per porta), formato fisso
  --auto         : prova tutti i FORMATI su una porta fissa (5s per formato)
  --single X     : manda solo il formato X continuamente (utile per conferma)
  --listen       : ascolta in ingresso qualsiasi messaggio OSC da Nuendo
  --burst X      : manda 20 messaggi istantanei del formato X (debug rapido)

Esempi:
  python scripts/osc_discovery.py --scan-ports
  python scripts/osc_discovery.py --auto --port 8000
  python scripts/osc_discovery.py --single D --port 8000
  python scripts/osc_discovery.py --listen --listen-port 9000
"""

import argparse
import math
import threading
import time
import logging

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("osc_discovery")

HOST = "127.0.0.1"

# ── porte candidate (tutte quelle che Nuendo può usare) ───────────────────────
CANDIDATE_PORTS = [8000, 8001, 9000, 9001, 7000, 7001, 6000, 5000, 5001, 4000, 3000, 10000]

# ── tutti i formati da testare ────────────────────────────────────────────────
# Ogni entry: id, descrizione stampata, template indirizzo, modalità argomenti
FORMATS = [
    # ── YOSC ──────────────────────────────────────────────────────────────────
    {"id": "A1", "desc": "/yosc/position  [n x y z]",        "addr": "/yosc/position",          "mode": "n+xyz"},
    {"id": "A2", "desc": "/yosc/position  [n x y z] (0-idx)","addr": "/yosc/position",          "mode": "n0+xyz"},
    {"id": "A3", "desc": "/yosc  [n x y z]",                 "addr": "/yosc",                   "mode": "n+xyz"},
    {"id": "A4", "desc": "/yosc/{n}  [x y z]",               "addr": "/yosc/{n}",               "mode": "xyz"},
    {"id": "A5", "desc": "/yosc/obj/{n}  [x y z]",           "addr": "/yosc/obj/{n}",           "mode": "xyz"},

    # ── ADM-OSC (EBU) ─────────────────────────────────────────────────────────
    {"id": "B1", "desc": "/adm/obj/{n}/cartesian  [x y z]",  "addr": "/adm/obj/{n}/cartesian",  "mode": "xyz_adm"},
    {"id": "B2", "desc": "/adm/obj/{n}/position  [x y z]",   "addr": "/adm/obj/{n}/position",   "mode": "xyz_adm"},
    {"id": "B3", "desc": "/adm/obj/{n}/x  val",              "addr": "/adm/obj/{n}/x",          "mode": "x_only_adm"},
    {"id": "B4", "desc": "/adm/obj/{n}/azimuth  deg",        "addr": "/adm/obj/{n}/azimuth",    "mode": "azimuth"},
    {"id": "B5", "desc": "/adm/obj/{n}/xyz  [x y z]",        "addr": "/adm/obj/{n}/xyz",        "mode": "xyz_adm"},

    # ── Steinberg/Nuendo generico ─────────────────────────────────────────────
    {"id": "C1", "desc": "/nuendo/obj/{n}/x  val",           "addr": "/nuendo/obj/{n}/x",       "mode": "x_only_norm"},
    {"id": "C2", "desc": "/nuendo/obj/{n}/pos  [x y z]",     "addr": "/nuendo/obj/{n}/pos",     "mode": "xyz"},
    {"id": "C3", "desc": "/steinberg/obj/{n}/x  val",        "addr": "/steinberg/obj/{n}/x",    "mode": "x_only_norm"},

    # ── Track-based ───────────────────────────────────────────────────────────
    {"id": "D1", "desc": "/track/{n}/xyz  [x y z]",          "addr": "/track/{n}/xyz",          "mode": "xyz"},
    {"id": "D2", "desc": "/track/{n}/x  val",                "addr": "/track/{n}/x",            "mode": "x_only_norm"},
    {"id": "D3", "desc": "/object/{n}/xyz  [x y z]",         "addr": "/object/{n}/xyz",         "mode": "xyz"},
    {"id": "D4", "desc": "/object/{n}/position  [x y z]",    "addr": "/object/{n}/position",    "mode": "xyz"},

    # ── Formati alternativi con spazio ────────────────────────────────────────
    {"id": "E1", "desc": "/audio/obj/{n}/x  val",            "addr": "/audio/obj/{n}/x",        "mode": "x_only_norm"},
    {"id": "E2", "desc": "/spatial/obj/{n}/pos  [x y z]",    "addr": "/spatial/obj/{n}/pos",    "mode": "xyz"},
]


def make_args(mode: str, x_norm: float, obj_n: int) -> list:
    """
    x_norm : 0.0 (sinistra) → 1.0 (destra)
    range Nuendo Stage: X/Y 0.0–1.0, ADM-OSC: -1.0–+1.0
    """
    if mode == "n+xyz":
        return [obj_n, x_norm, 0.5, 0.0]
    if mode == "n0+xyz":          # oggetti 0-indexed
        return [obj_n - 1, x_norm, 0.5, 0.0]
    if mode == "xyz":
        return [x_norm, 0.5, 0.0]
    if mode == "xyz_adm":         # ADM: -1/+1
        return [x_norm * 2.0 - 1.0, 0.0, 0.0]
    if mode == "x_only_adm":
        return [x_norm * 2.0 - 1.0]
    if mode == "x_only_norm":
        return [x_norm]
    if mode == "azimuth":         # ADM: -180/+180 deg
        return [(x_norm - 0.5) * 180.0]
    return [x_norm, 0.5, 0.0]


def oscillate(elapsed: float) -> float:
    """Sinusoide lenta 0→1→0, periodo 3 secondi — molto visibile."""
    return (math.sin(elapsed * 2 * math.pi / 3.0) + 1.0) / 2.0


# ── Invio continuo per una combinazione formato+porta ─────────────────────────

def send_format_continuous(client, fmt: dict, duration_s: float = 6.0, obj_range=(1, 4)):
    t0 = time.perf_counter()
    log.info("━" * 58)
    log.info(f"  FORMATO {fmt['id']:4s} | {fmt['desc']}")
    log.info(f"  porta attuale — invia per {duration_s}s — GUARDA NUENDO")
    log.info("━" * 58)

    while time.perf_counter() - t0 < duration_s:
        elapsed = time.perf_counter() - t0
        x = oscillate(elapsed)
        for n in range(obj_range[0], obj_range[1] + 1):
            addr = fmt["addr"].replace("{n}", str(n))
            args = make_args(fmt["mode"], x, n)
            client.send_message(addr, args)
            time.sleep(0.005)
        time.sleep(0.03)   # ~25 Hz

    log.info(f"  Fine {fmt['id']}\n")


# ── modalità: burst istantaneo (debug) ────────────────────────────────────────

def run_burst(host: str, port: int, fmt_id: str, n_msgs: int = 20):
    client = udp_client.SimpleUDPClient(host, port)
    fmt = next((f for f in FORMATS if f["id"].upper() == fmt_id.upper()), None)
    if not fmt:
        log.error(f"Formato '{fmt_id}' non trovato.")
        return
    log.info(f"BURST {n_msgs}×  {fmt['id']} → {host}:{port}")
    for i in range(n_msgs):
        x = i / (n_msgs - 1)
        for n in range(1, 5):
            addr = fmt["addr"].replace("{n}", str(n))
            args = make_args(fmt["mode"], x, n)
            client.send_message(addr, args)
            log.info(f"  {addr}  {args}")
        time.sleep(0.05)


# ── modalità: scansione di tutte le porte con un formato fisso ───────────────

def run_scan_ports(host: str, fmt_id: str = "A1", duration_s: float = 5.0):
    fmt = next((f for f in FORMATS if f["id"].upper() == fmt_id.upper()), FORMATS[0])
    log.info(f"\n🔍 SCAN PORTE  ({len(CANDIDATE_PORTS)} porte, {duration_s}s ciascuna)")
    log.info(f"   Formato fisso: {fmt['id']} — {fmt['desc']}\n")

    for port in CANDIDATE_PORTS:
        client = udp_client.SimpleUDPClient(host, port)
        log.info(f">>> PORTA {port}")
        send_format_continuous(client, fmt, duration_s)

    log.info("✅ Scan porte completato — dimmi su quale porta hai visto movimento!")


# ── modalità: tutti i formati su una porta fissa ─────────────────────────────

def run_auto(host: str, port: int, duration_s: float = 5.0):
    client = udp_client.SimpleUDPClient(host, port)
    log.info(f"\n🔍 AUTO (tutti i formati) → {host}:{port}")
    log.info(f"   {len(FORMATS)} formati × {duration_s}s ciascuno\n")

    for fmt in FORMATS:
        send_format_continuous(client, fmt, duration_s)

    log.info("✅ Completato — dimmi l'id del formato che ha mosso gli oggetti!")


# ── modalità: singolo formato ─────────────────────────────────────────────────

def run_single(host: str, port: int, fmt_id: str, duration_s: float = 15.0):
    client = udp_client.SimpleUDPClient(host, port)
    fmt = next((f for f in FORMATS if f["id"].upper() == fmt_id.upper()), None)
    if not fmt:
        log.error(f"Formato '{fmt_id}' non trovato. Disponibili: {[f['id'] for f in FORMATS]}")
        return
    send_format_continuous(client, fmt, duration_s)


# ── modalità: listener passivo (vede se Nuendo manda qualcosa) ────────────────

def run_listen(listen_port: int = 9000, duration_s: float = 30.0):
    received = []

    def catch_all(addr, *args):
        msg = f"{addr}  {list(args)}"
        received.append(msg)
        log.info(f"◀ RICEVUTO: {msg}")

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(catch_all)

    log.info(f"\n👂 In ascolto su 0.0.0.0:{listen_port}  ({duration_s}s)")
    log.info("   Avvia qualcosa in Nuendo (play, move, etc.) per vedere i messaggi\n")

    server = BlockingOSCUDPServer(("0.0.0.0", listen_port), dispatcher)
    server.socket.settimeout(duration_s)

    deadline = time.perf_counter() + duration_s
    try:
        while time.perf_counter() < deadline:
            server.handle_request()
    except Exception:
        pass

    if received:
        log.info(f"\n✅ Ricevuti {len(received)} messaggi da Nuendo!")
    else:
        log.info("\n⚠️  Nessun messaggio ricevuto — Nuendo non stava mandando nulla.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OSC Discovery v2 — trova il formato/porta giusto per Nuendo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # scansiona tutte le porte (formato A1 fisso):
  python scripts/osc_discovery.py --scan-ports

  # prova tutti i formati su porta 8000:
  python scripts/osc_discovery.py --auto --port 8000

  # manda solo il formato B1 su porta 8000 per 20s:
  python scripts/osc_discovery.py --single B1 --port 8000 --duration 20

  # manda 20 messaggi istantanei del formato A1 (debug):
  python scripts/osc_discovery.py --burst A1 --port 8000

  # ascolta su porta 9000 per 30s (vedi se Nuendo parla):
  python scripts/osc_discovery.py --listen --listen-port 9000
""")
    parser.add_argument("--scan-ports",  action="store_true",
                        help="Prova tutte le porte candidate con formato A1")
    parser.add_argument("--auto",        action="store_true",
                        help="Prova tutti i formati su --port")
    parser.add_argument("--single",      metavar="ID",
                        help="Manda solo il formato ID (es. A1, B1…)")
    parser.add_argument("--burst",       metavar="ID",
                        help="Manda 20 messaggi istantanei del formato ID")
    parser.add_argument("--listen",      action="store_true",
                        help="Ascolta messaggi OSC in arrivo da Nuendo")
    parser.add_argument("--port",        type=int, default=8000,
                        help="Porta Nuendo (default 8000)")
    parser.add_argument("--host",        default=HOST,
                        help="Host Nuendo (default 127.0.0.1)")
    parser.add_argument("--listen-port", type=int, default=9000,
                        help="Porta locale per --listen (default 9000)")
    parser.add_argument("--duration",    type=float, default=5.0,
                        help="Secondi per formato (default 5)")
    parser.add_argument("--scan-fmt",    default="A1",
                        help="Formato da usare con --scan-ports (default A1)")
    args = parser.parse_args()

    if args.scan_ports:
        run_scan_ports(args.host, args.scan_fmt, args.duration)
    elif args.auto:
        run_auto(args.host, args.port, args.duration)
    elif args.single:
        run_single(args.host, args.port, args.single, args.duration)
    elif args.burst:
        run_burst(args.host, args.port, args.burst)
    elif args.listen:
        run_listen(args.listen_port, duration_s=30.0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

