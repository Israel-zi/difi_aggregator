"""
sender.py
---------
DIFI Unified Sender module.

Reads (context_bytes, data_bytes) tuples from the Packetizer output queue
and transmits them over UDP unicast to the DIFI RF Frontend.

For the PoC, the "RF Frontend" is the Receiver module listening on port 50010.
"""

import socket
import threading
import time

from modules.packetizer import Packetizer


# ─────────────────────────────────────────────
# Sender
# ─────────────────────────────────────────────

class DifiSender:
    """
    Transmits unified DIFI packets (Context + Data) to a single destination
    over UDP unicast.

    Parameters
    ----------
    packetizer  : Packetizer instance to read from
    dest_host   : destination IP address (RF Frontend)
    dest_port   : destination UDP port
    """

    def __init__(
        self,
        packetizer: Packetizer,
        dest_host: str = "127.0.0.1",
        dest_port: int = 50010,
    ):
        self._packetizer  = packetizer
        self._dest        = (dest_host, dest_port)
        self._sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_evt    = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="sender"
        )

        # stats
        self.packets_sent    = 0
        self.bytes_sent      = 0
        self.context_sent    = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._thread.start()
        print(f"[Sender] Started | dest={self._dest[0]}:{self._dest[1]}")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        self._sock.close()
        print(
            f"[Sender] Stopped | "
            f"packets={self.packets_sent} | "
            f"bytes={self.bytes_sent}"
        )

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            item = self._packetizer.get(timeout=0.05)
            if item is None:
                continue

            ctx_bytes, data_bytes = item

            # send Context packet first if present
            if ctx_bytes:
                self._sock.sendto(ctx_bytes, self._dest)
                self.context_sent += 1
                self.bytes_sent   += len(ctx_bytes)

            # send Data packet
            self._sock.sendto(data_bytes, self._dest)
            self.packets_sent += 1
            self.bytes_sent   += len(data_bytes)

            if self.packets_sent % 100 == 0:
                print(
                    f"  [Sender] sent {self.packets_sent} data pkts | "
                    f"{self.bytes_sent / 1024:.1f} KB total"
                )