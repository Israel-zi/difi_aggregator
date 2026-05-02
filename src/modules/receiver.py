"""
receiver.py
-----------
DIFI Receiver module.

Listens on UDP port 50010 for the unified DIFI stream emitted by the Sender,
parses incoming Data + Context packets, reconstructs the IQ signal, and
displays a live spectrum for verification.

This confirms that the aggregated stream contains the carriers from
both original Generator streams.
"""

import socket
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    PACKET_TYPE_DATA,
    PACKET_TYPE_CONTEXT,
)


# ─────────────────────────────────────────────
# Receiver
# ─────────────────────────────────────────────

class DifiReceiver:
    """
    Receives the unified DIFI stream and stores IQ data for display/analysis.

    Parameters
    ----------
    host        : local address to bind
    port        : UDP port to listen on
    buffer_size : number of IQ samples to keep in the rolling display buffer
    """

    MAX_UDP_SIZE = 65535

    def __init__(
        self,
        host: str       = "0.0.0.0",
        port: int       = 50010,
        buffer_size: int = 8192,
    ):
        self._host        = host
        self._port        = port
        self._buffer_size = buffer_size
        self._sock        = None
        self._stop_evt    = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="receiver"
        )

        # rolling IQ buffer (latest N samples)
        self._iq_buffer   = np.zeros(buffer_size, dtype=np.complex64)
        self._lock        = threading.Lock()

        # latest context
        self.context      = None

        # stats
        self.data_received    = 0
        self.context_received = 0
        self.parse_errors     = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self._host, self._port))
        self._thread.start()
        print(f"[Receiver] Listening on {self._host}:{self._port}")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        if self._sock:
            self._sock.close()
        print(
            f"[Receiver] Stopped | "
            f"data={self.data_received} ctx={self.context_received}"
        )

    # ── data access ────────────────────────────────────────────────────────

    def get_iq_snapshot(self) -> np.ndarray:
        """Return a copy of the current IQ buffer (thread-safe)."""
        with self._lock:
            return self._iq_buffer.copy()

    def get_sample_rate(self) -> float:
        """Return sample rate from the latest context packet (or default)."""
        if self.context:
            return self.context.sample_rate_hz
        return 48_000.0

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._handle(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle(self, data: bytes):
        if len(data) < 4:
            return

        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF

        try:
            if pkt_type == PACKET_TYPE_DATA:
                bit_depth = self.context.sample_bit_depth if self.context else 16
                pkt = DifiDataPacket.from_bytes(data, sample_bit_depth=bit_depth)
                self._update_buffer(pkt.payload)
                self.data_received += 1

            elif pkt_type == PACKET_TYPE_CONTEXT:
                pkt          = DifiContextPacket.from_bytes(data)
                self.context = pkt
                self.context_received += 1

        except (ValueError, Exception) as exc:
            self.parse_errors += 1
            print(f"[Receiver] Parse error: {exc}")

    def _update_buffer(self, new_samples: np.ndarray):
        """Append new samples to the rolling buffer."""
        n = len(new_samples)
        with self._lock:
            if n >= self._buffer_size:
                self._iq_buffer[:] = new_samples[-self._buffer_size:]
            else:
                self._iq_buffer = np.roll(self._iq_buffer, -n)
                self._iq_buffer[-n:] = new_samples


# ─────────────────────────────────────────────
# Live spectrum display
# ─────────────────────────────────────────────

def run_spectrum_display(receiver: DifiReceiver):
    """Display a live FFT spectrum of the received aggregated DIFI stream."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("DIFI Receiver — Aggregated Stream Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-80, 10)
    ax.grid(True, alpha=0.4)

    line, = ax.plot([], [], color="cyan", linewidth=1.2)
    info_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        color="white"
    )

    def update(_frame):
        iq      = receiver.get_iq_snapshot()
        fs      = receiver.get_sample_rate()
        n       = len(iq)

        window  = np.hanning(n)
        X       = np.fft.fftshift(np.fft.fft(iq * window))
        freqs   = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
        mag_db  = 20 * np.log10(np.abs(X) / n + 1e-12)

        line.set_data(freqs, mag_db)
        ax.set_xlim(freqs[0], freqs[-1])

        ctx = receiver.context
        info = (
            f"Data pkts : {receiver.data_received}\n"
            f"Ctx pkts  : {receiver.context_received}\n"
            f"Fs        : {fs:.0f} Hz"
        )
        if ctx:
            info += f"\nRF freq : {ctx.rf_ref_freq_hz / 1e6:.3f} MHz"
        info_text.set_text(info)

        return line, info_text

    ani = animation.FuncAnimation(
        fig, update, interval=100, blit=True, cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    receiver = DifiReceiver(port=50010)
    receiver.start()

    print("\n[Receiver] Waiting for unified DIFI stream on port 50010 ...")
    print("           Run main.py to start the full pipeline.")
    print("           Press Ctrl+C to stop.\n")

    try:
        run_spectrum_display(receiver)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()