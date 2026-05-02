"""
input_capture.py
----------------
DIFI Input Capture module.

Listens on multiple UDP ports simultaneously (one per Generator/Modem),
parses incoming DIFI packets (Context + Data), tags them with their
source stream ID, and places them into a shared queue for the Aggregator.

Each port corresponds to one DIFI stream:
  Port 50001 -> Stream ID 0x00000001 (Modem 1)
  Port 50002 -> Stream ID 0x00000002 (Modem 2)
"""

import socket
import struct
import threading
import queue
import time
from dataclasses import dataclass

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    PACKET_TYPE_DATA,
    PACKET_TYPE_CONTEXT,
)


# ─────────────────────────────────────────────
# Tagged packet container
# ─────────────────────────────────────────────

@dataclass
class CapturedPacket:
    """A received DIFI packet tagged with its source port."""
    source_port:    int
    received_at:    float                           # time.monotonic() timestamp
    packet:         DifiDataPacket | DifiContextPacket


# ─────────────────────────────────────────────
# Per-port listener thread
# ─────────────────────────────────────────────

class PortListener(threading.Thread):
    """
    Listens on a single UDP port, parses DIFI packets, and puts them
    into the shared output queue.
    """

    MAX_UDP_SIZE = 65535

    def __init__(
        self,
        port: int,
        out_queue: queue.Queue,
        host: str = "0.0.0.0",
        timeout: float = 1.0,
    ):
        super().__init__(daemon=True, name=f"listener-{port}")
        self.port      = port
        self.out_queue = out_queue
        self.host      = host
        self.timeout   = timeout
        self._stop_evt = threading.Event()
        self._sock     = None

        # per-stream statistics
        self.stats = {
            "data_received":    0,
            "context_received": 0,
            "parse_errors":     0,
        }

    def stop(self):
        self._stop_evt.set()
        # Close socket immediately to unblock recvfrom — no 1-second timeout wait
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(self.timeout)
        self._sock.bind((self.host, self.port))

        print(f"[Capture] Listening on {self.host}:{self.port}")

        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._parse_and_enqueue(data)
            except socket.timeout:
                continue
            except OSError:
                break   # socket closed (by stop() or network error)

        try:
            self._sock.close()
        except OSError:
            pass  # already closed by stop()
        print(f"[Capture] Port {self.port} listener stopped")

    def _parse_and_enqueue(self, data: bytes):
        """Detect packet type from header and parse accordingly."""
        if len(data) < 4:
            return

        # peek at bits 31-28 of Word 1 to detect packet type
        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF

        try:
            if pkt_type == PACKET_TYPE_DATA:
                pkt = DifiDataPacket.from_bytes(data)
                self.stats["data_received"] += 1

            elif pkt_type == PACKET_TYPE_CONTEXT:
                pkt = DifiContextPacket.from_bytes(data)
                self.stats["context_received"] += 1

            else:
                # unknown packet type — skip silently
                return

            self.out_queue.put(CapturedPacket(
                source_port = self.port,
                received_at = time.monotonic(),
                packet      = pkt,
            ))

        except (ValueError, struct.error) as exc:
            self.stats["parse_errors"] += 1
            print(f"[Capture] Parse error on port {self.port}: {exc}")


# ─────────────────────────────────────────────
# Multi-port capture manager
# ─────────────────────────────────────────────

class InputCapture:
    """
    Manages multiple PortListener threads and exposes a single queue
    containing tagged packets from all sources.

    Usage
    -----
        capture = InputCapture(ports=[50001, 50002])
        capture.start()

        while True:
            captured = capture.get(timeout=1.0)
            if captured:
                process(captured.packet)

        capture.stop()
    """

    def __init__(
        self,
        ports: list,
        host: str        = "0.0.0.0",
        queue_maxsize: int = 1000,
    ):
        self._out_queue = queue.Queue(maxsize=queue_maxsize)
        self._listeners = [
            PortListener(port=p, out_queue=self._out_queue, host=host)
            for p in ports
        ]

    def start(self):
        """Start all listener threads."""
        for listener in self._listeners:
            listener.start()
        print(f"[Capture] Started {len(self._listeners)} listener(s)")

    def stop(self):
        """Stop all listener threads."""
        for listener in self._listeners:
            listener.stop()
        for listener in self._listeners:
            listener.join(timeout=2.0)
        print("[Capture] All listeners stopped")

    def get(self, timeout: float = 0.1):
        """
        Retrieve the next captured packet.
        Returns None if nothing arrived within `timeout` seconds.
        """
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> dict:
        """Return combined statistics from all listeners."""
        combined = {"data_received": 0, "context_received": 0, "parse_errors": 0}
        for listener in self._listeners:
            for key in combined:
                combined[key] += listener.stats[key]
        return combined

    @property
    def queue_size(self) -> int:
        return self._out_queue.qsize()


# ─────────────────────────────────────────────
# Quick self-test (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    TEST_PORTS = [50001, 50002]
    capture    = InputCapture(ports=TEST_PORTS)
    capture.start()

    print(f"\n[Test] Waiting for DIFI packets on ports {TEST_PORTS} ...")
    print("       Run generator.py in another terminal to send packets.")
    print("       Press Ctrl+C to stop.\n")

    received = 0
    try:
        while True:
            captured = capture.get(timeout=1.0)
            if captured:
                received += 1
                pkt = captured.packet
                print(
                    f"  Port {captured.source_port} | "
                    f"type={'DATA' if isinstance(pkt, DifiDataPacket) else 'CTX'} | "
                    f"stream=0x{pkt.stream_id:08X} | "
                    f"seq={pkt.seq_num}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print(f"\n[Test] Received {received} packets total")
        print(f"[Test] Stats: {capture.stats()}")