"""mtc_sync.py — MTC (MIDI Time Code) listener for Nuendo → Python sync.

Riceve i Quarter Frame MTC da IAC Bus 1 (Nuendo = Master),
ricostruisce il timecode corrente e triggera gli eventi OSC
della spatial timeline al momento giusto.

Flusso:
    Nuendo play  →  MTC QF su IAC Bus 1
    mtc_sync     →  decodifica timecode  →  calcola delta_s
                 →  dispatcha eventi OSC alla posizione corretta

Utilizzo:
    python src/cre/mtc_sync.py --score data/musicxml/quartet.musicxml
    python src/cre/mtc_sync.py --score data/musicxml/quartet.musicxml --fps 25 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time
import threading
from pathlib import Path
from typing import Optional

import pygame.midi
import numpy as np
import pandas as pd
from pythonosc.udp_client import SimpleUDPClient

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────────────────────
# MTC QUARTER FRAME DECODER
# ──────────────────────────────────────────────

class MTCDecoder:
    """Ricostruisce il timecode HH:MM:SS:FF dai 8 Quarter Frame MTC.

    MTC usa 8 messaggi Quarter Frame consecutivi per trasmettere
    un singolo frame. Ogni QF porta 4 bit (nibble) di un campo TC.

    Byte QF: 0xF1 + data_byte
    data_byte: [tipo (3 bit)] [valore (4 bit)]
        tipo 0 = frame LS nibble
        tipo 1 = frame MS nibble
        tipo 2 = seconds LS nibble
        tipo 3 = seconds MS nibble
        tipo 4 = minutes LS nibble
        tipo 5 = minutes MS nibble
        tipo 6 = hours LS nibble
        tipo 7 = hours MS nibble + frame rate
    """

    FRAME_RATES = {0: 24, 1: 25, 2: 29.97, 3: 30}

    def __init__(self) -> None:
        self._nibbles = [0] * 8          # 8 nibble, uno per QF
        self._count = 0                  # quanti QF ricevuti finora
        self._complete = False           # True quando abbiamo tutti e 8
        self.hours = 0
        self.minutes = 0
        self.seconds = 0
        self.frames = 0
        self.fps = 25.0
        # Callback chiamata ogni volta che il TC è aggiornato
        self.on_timecode: Optional[callable] = None

    def feed_quarter_frame(self, data_byte: int) -> None:
        """Processa un singolo byte Quarter Frame (senza il 0xF1 iniziale)."""
        piece_type = (data_byte >> 4) & 0x07
        value      = data_byte & 0x0F
        self._nibbles[piece_type] = value

        # Aggiorniamo il TC completo ogni 8 messaggi (ogni 2 frame a 25fps)
        self._count += 1
        if self._count >= 8:
            self._count = 0
            self._complete = True
            self._rebuild()

    def feed_full_frame(self, data: bytes) -> None:
        """Processa un Full Frame MTC SysEx (0xF0 7F 7F 01 01 hh mm ss ff F7)."""
        if len(data) >= 8:
            hh = data[5]
            mm = data[6]
            ss = data[7]
            ff = data[8] if len(data) > 8 else 0
            rate_bits = (hh >> 5) & 0x03
            self.fps     = self.FRAME_RATES.get(rate_bits, 25.0)
            self.hours   = hh & 0x1F
            self.minutes = mm & 0x3F
            self.seconds = ss & 0x3F
            self.frames  = ff & 0x1F
            self._complete = True
            self._notify()

    def _rebuild(self) -> None:
        n = self._nibbles
        self.frames  = n[0] | (n[1] << 4)
        self.seconds = n[2] | (n[3] << 4)
        self.minutes = n[4] | (n[5] << 4)
        rate_bits    = (n[7] >> 1) & 0x03
        self.hours   = n[6] | ((n[7] & 0x01) << 4)
        self.fps     = self.FRAME_RATES.get(rate_bits, 25.0)
        self._notify()

    def _notify(self) -> None:
        if self.on_timecode:
            self.on_timecode(self.to_seconds(), self)

    def to_seconds(self) -> float:
        """Converte il TC corrente in secondi assoluti."""
        return (self.hours * 3600
                + self.minutes * 60
                + self.seconds
                + self.frames / self.fps)

    def __str__(self) -> str:
        return (f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:"
                f"{self.frames:02d}  @{self.fps}fps  = {self.to_seconds():.3f}s")


# ──────────────────────────────────────────────
# MTC-DRIVEN OSC SCHEDULER
# ──────────────────────────────────────────────

class MTCOscScheduler:
    """Dispatcha eventi OSC sincronizzati con il timecode MTC di Nuendo.

    Legge il TC corrente da Nuendo via IAC Bus 1 e invia ogni evento
    OSC esattamente quando il TC corrisponde al timestamp della timeline.
    """

    def __init__(
        self,
        timeline: pd.DataFrame,
        osc_client: Optional[SimpleUDPClient],
        midi_port_name: str = "Driver IAC IAC Bus 1",
        fps: float = 25.0,
        dry_run: bool = False,
    ) -> None:
        self.timeline    = timeline.sort_values("timestamp_s").reset_index(drop=True)
        self.osc_client  = osc_client
        self.port_name   = midi_port_name
        self.fps         = fps
        self.dry_run     = dry_run
        self._decoder    = MTCDecoder()
        self._stop_event = threading.Event()
        self._tc_seconds = 0.0
        self._tc_lock    = threading.Lock()
        self._midi_port_idx: Optional[int] = None

        self._decoder.on_timecode = self._on_timecode_update

    # ── Callback MTC ──────────────────────────────────────────────────────

    def _midi_poll_loop(self, midi_in: "pygame.midi.Input") -> None:
        """Thread dedicato: legge tutti i messaggi MIDI e li invia al decoder."""
        while not self._stop_event.is_set():
            if midi_in.poll():
                events = midi_in.read(64)
                for event in events:
                    data, _ts = event
                    status = data[0]
                    if status == 0xF1:
                        self._decoder.feed_quarter_frame(data[1])
                    elif status == 0xF0:
                        self._decoder.feed_full_frame(bytes(data))
            time.sleep(0.0005)  # poll ogni 0.5ms

    def _on_timecode_update(self, tc_s: float, decoder: MTCDecoder) -> None:
        with self._tc_lock:
            self._tc_seconds = tc_s
        LOGGER.debug("MTC: %s", decoder)

    # ── Scheduler loop ─────────────────────────────────────────────────────

    def _dispatch_loop(self, start_tc: float = 0.0) -> None:
        """Loop principale: scorre la timeline e aspetta il TC giusto.

        Args:
            start_tc: Timecode (secondi) al momento di avvio del dispatch.
                      Tutti gli eventi con timestamp_s < start_tc vengono
                      saltati, così il sistema funziona anche se la riproduzione
                      parte da una posizione diversa dalla battuta 1.
        """
        # Fast-forward: trova il primo evento da inviare
        timestamps = self.timeline["timestamp_s"].values
        next_idx = int(np.searchsorted(timestamps, start_tc, side="left"))

        if next_idx > 0:
            LOGGER.info(
                "Playback inizia a TC=%.3fs — saltati %d eventi precedenti, "
                "primo evento = index %d @ %.3fs",
                start_tc, next_idx, next_idx,
                float(self.timeline.iloc[next_idx]["timestamp_s"]) if next_idx < len(self.timeline) else float("inf"),
            )
        else:
            LOGGER.info("Playback dalla battuta 1 — %d eventi da inviare", len(self.timeline))

        n_events = len(self.timeline)
        LOGGER.info("Scheduler MTC pronto — %d eventi rimasti da inviare", n_events - next_idx)

        prev_tc = start_tc  # usato per rilevare seek/rewind

        while not self._stop_event.is_set() and next_idx < n_events:
            row = self.timeline.iloc[next_idx]
            target_tc = float(row["timestamp_s"])

            # Aspetta che il TC superi il target
            while not self._stop_event.is_set():
                with self._tc_lock:
                    current_tc = self._tc_seconds

                # ── Seek / rewind detection ──────────────────────────────
                # Se il TC è sceso significativamente rispetto all'ultimo valore
                # noto, l'utente ha spostato la testina all'indietro: ricalcoliamo
                # next_idx dalla nuova posizione.
                if current_tc < prev_tc - 0.5:
                    LOGGER.warning(
                        "Seek all'indietro rilevato: %.3fs → %.3fs — ricalcolo posizione",
                        prev_tc, current_tc,
                    )
                    next_idx = int(np.searchsorted(timestamps, current_tc, side="left"))
                    prev_tc = current_tc
                    break  # ricomincia il while esterno con il nuovo next_idx

                prev_tc = current_tc

                if current_tc >= target_tc:
                    break
                time.sleep(0.001)   # poll ogni 1ms

            else:
                # _stop_event set nel loop interno
                break

            if self._stop_event.is_set():
                break

            # Ricontrolla che l'indice non sia già stato aggiornato da un seek
            if next_idx >= n_events:
                break
            row = self.timeline.iloc[next_idx]
            target_tc = float(row["timestamp_s"])
            with self._tc_lock:
                current_tc = self._tc_seconds
            if current_tc >= target_tc:
                self._dispatch_row(row)
                next_idx += 1

        LOGGER.info("Scheduler MTC completato (%d eventi inviati).", next_idx)

    def _dispatch_row(self, row: pd.Series) -> None:
        idx   = int(row["object_index"])
        x_adm = max(-1.0, min(1.0, float(row["x"])))
        y_adm = max(-1.0, min(1.0, float(row["y"])))
        z_adm = max(0.0,  min(1.0, float(row["z"])))
        addr  = f"/adm/obj/{idx}/xyz"
        args  = [x_adm, y_adm, z_adm]

        tc_now = self._tc_seconds
        if self.dry_run or not self.osc_client:
            LOGGER.info("DRY  TC=%.3fs  %s -> [%.3f, %.3f, %.3f]",
                        tc_now, addr, *args)
        else:
            self.osc_client.send_message(addr, args)
            LOGGER.debug("SENT TC=%.3fs  %s -> [%.3f, %.3f, %.3f]",
                         tc_now, addr, *args)

    # ── Start / Stop ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Apre la porta MIDI via pygame.midi e avvia i thread."""
        pygame.midi.init()
        n_ports = pygame.midi.get_count()
        LOGGER.info("Porte MIDI disponibili: %d", n_ports)

        port_idx = None
        for i in range(n_ports):
            info = pygame.midi.get_device_info(i)
            _interf, name, is_input, _is_output, _opened = info
            name_str = name.decode()
            if is_input and ("IAC" in name_str or self.port_name in name_str):
                port_idx = i
                LOGGER.info("Porta MIDI selezionata: [%d] %s", i, name_str)
                break

        if port_idx is None:
            raise RuntimeError(
                f"Porta MIDI IAC non trovata. "
                f"Porte disponibili: {[pygame.midi.get_device_info(i)[1].decode() for i in range(n_ports)]}"
            )

        midi_in = pygame.midi.Input(port_idx)
        LOGGER.info("Ascoltando MTC su porta %d", port_idx)
        LOGGER.info("Premi PLAY in Nuendo per iniziare…")

        # Thread di polling MIDI
        poll_thread = threading.Thread(
            target=self._midi_poll_loop, args=(midi_in,), daemon=True
        )
        poll_thread.start()

        # Aspetta il primo TC valido e registra la posizione di partenza
        start_tc = self._wait_for_first_tc()

        # Thread di dispatch OSC (parte dal TC corrente)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, args=(start_tc,), daemon=True
        )
        self._dispatch_thread.start()

        # Tieni i riferimenti per lo stop
        self._midi_in_ref = midi_in
        self._poll_thread = poll_thread

    def _wait_for_first_tc(self) -> float:
        """Blocca finché non riceve almeno un TC valido da Nuendo.

        Returns:
            Il valore di timecode (secondi) al momento del primo lock.
        """
        LOGGER.info("In attesa del primo timecode MTC…")
        while not self._stop_event.is_set():
            with self._tc_lock:
                tc = self._tc_seconds
            if self._decoder._complete:
                LOGGER.info("Primo TC ricevuto: %.3fs — dispatch avviato", tc)
                return tc
            time.sleep(0.01)
        return 0.0

    def join(self, timeout: float = 3600.0) -> None:
        """Aspetta la fine del dispatch."""
        if hasattr(self, "_dispatch_thread"):
            self._dispatch_thread.join(timeout=timeout)

    def stop(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_midi_in_ref"):
            self._midi_in_ref.close()
        pygame.midi.quit()
        LOGGER.info("MTCOscScheduler fermato.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MTC-synced OSC dispatcher per Nuendo ADM objects",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--score",    required=True, help="Path al file MusicXML")
    p.add_argument("--fps",      type=float, default=25.0, help="Frame rate MTC (default: 25)")
    p.add_argument("--port",     default="Driver IAC IAC Bus 1", help="Nome porta MIDI IAC")
    p.add_argument("--osc-host", default="127.0.0.1")
    p.add_argument("--osc-port", type=int, default=8000)
    p.add_argument("--dry-run",  action="store_true", help="Non invia OSC, solo log")
    p.add_argument("--rules-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import inline — aggiunge src/ al path per trovare rule_engine
    import sys
    src_path = Path(__file__).resolve().parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from cre.rule_engine import (
        converter, extract_chord_events, extract_dynamic_events,
        build_spatial_timeline, load_rule, load_all_rules, load_json
    )

    pipeline_root = Path(__file__).resolve().parents[2]
    score_path    = Path(args.score).expanduser()
    if not score_path.is_absolute():
        score_path = pipeline_root / score_path

    LOGGER.info("Caricamento partitura: %s", score_path)
    score_obj      = converter.parse(str(score_path))
    chord_events   = extract_chord_events(score_obj)
    dynamic_events = extract_dynamic_events(score_obj)

    if args.rules_dir:
        rules = load_all_rules(Path(args.rules_dir))
    else:
        rule_path = pipeline_root / "configs" / "rules" / "quartet_harmony_rule.json"
        rules = [load_rule(rule_path)]

    LOGGER.info("Building spatial timeline…")
    timeline_df = build_spatial_timeline(chord_events, dynamic_events, rules)
    LOGGER.info("Timeline: %d eventi, durata %.1fs",
                len(timeline_df), timeline_df["timestamp_s"].max())

    osc_client = None if args.dry_run else SimpleUDPClient(args.osc_host, args.osc_port)
    if not args.dry_run:
        LOGGER.info("OSC target → %s:%d", args.osc_host, args.osc_port)

    scheduler = MTCOscScheduler(
        timeline   = timeline_df,
        osc_client = osc_client,
        midi_port_name = args.port,
        fps        = args.fps,
        dry_run    = args.dry_run,
    )

    try:
        scheduler.start()
        scheduler.join()
    except KeyboardInterrupt:
        LOGGER.info("Interrotto dall'utente.")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
