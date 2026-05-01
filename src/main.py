"""
main.py
-------
DIFI Aggregator PoC — Full Pipeline Orchestrator.

Starts all modules in the correct order and runs the pipeline:

  Generator 1 (tone=2000Hz, port=50001)  ──┐
                                             ├─> InputCapture ─> Aggregator ─> Packetizer ─> Sender ─> Receiver (port=50010)
  Generator 2 (tone=6000Hz, port=50002)  ──┘

Press Ctrl+C to stop all modules cleanly.
"""

import threading
import time
import sys

from modules.generator    import DifiGenerator
from modules.input_capture import InputCapture
from modules.aggregator   import Aggregator
from modules.packetizer   import Packetizer
from modules.sender       import DifiSender
from modules.receiver     import DifiReceiver, run_spectrum_display


# ─────────────────────────────────────────────
# Pipeline configuration
# ─────────────────────────────────────────────

SAMPLE_RATE_HZ    = 48_000
SAMPLES_PER_PKT   = 1024
BIT_DEPTH         = 16
RF_FREQ_HZ        = 437_000_000
PACKET_RATE_HZ    = SAMPLE_RATE_HZ / SAMPLES_PER_PKT   # ~46.875 Hz

GENERATOR_1 = dict(
    stream_id = 0x00000001,
    tone_hz   = 2_000.0,
    dest_port = 50001,
)

GENERATOR_2 = dict(
    stream_id = 0x00000002,
    tone_hz   = 6_000.0,
    dest_port = 50002,
)

CAPTURE_PORTS     = [50001, 50002]
EXPECTED_STREAMS  = [0x00000001, 0x00000002]
RECEIVER_PORT     = 50010


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DIFI Aggregator PoC — Starting pipeline")
    print("=" * 60)

    # ── 1. Generators ──────────────────────────────────────────────
    gen1 = DifiGenerator(
        stream_id      = GENERATOR_1["stream_id"],
        tone_hz        = GENERATOR_1["tone_hz"],
        dest_port      = GENERATOR_1["dest_port"],
        sample_rate_hz = SAMPLE_RATE_HZ,
        samples_per_pkt = SAMPLES_PER_PKT,
        bit_depth      = BIT_DEPTH,
        rf_ref_freq_hz = RF_FREQ_HZ,
    )

    gen2 = DifiGenerator(
        stream_id      = GENERATOR_2["stream_id"],
        tone_hz        = GENERATOR_2["tone_hz"],
        dest_port      = GENERATOR_2["dest_port"],
        sample_rate_hz = SAMPLE_RATE_HZ,
        samples_per_pkt = SAMPLES_PER_PKT,
        bit_depth      = BIT_DEPTH,
        rf_ref_freq_hz = RF_FREQ_HZ,
    )

    gen1_thread = threading.Thread(
        target=gen1.run,
        kwargs=dict(num_packets=0, packet_rate_hz=PACKET_RATE_HZ),
        daemon=True, name="gen1"
    )
    gen2_thread = threading.Thread(
        target=gen2.run,
        kwargs=dict(num_packets=0, packet_rate_hz=PACKET_RATE_HZ),
        daemon=True, name="gen2"
    )

    # ── 2. Input Capture ───────────────────────────────────────────
    capture = InputCapture(ports=CAPTURE_PORTS)

    # ── 3. Aggregator ──────────────────────────────────────────────
    aggregator = Aggregator(
        capture          = capture,
        expected_streams = EXPECTED_STREAMS,
        chunk_size       = SAMPLES_PER_PKT,
    )

    # ── 4. Packetizer ──────────────────────────────────────────────
    packetizer = Packetizer(aggregator=aggregator)

    # ── 5. Sender ──────────────────────────────────────────────────
    sender = DifiSender(packetizer=packetizer, dest_port=RECEIVER_PORT)

    # ── 6. Receiver ────────────────────────────────────────────────
    receiver = DifiReceiver(port=RECEIVER_PORT)

    # ── Start pipeline (order matters) ─────────────────────────────
    print("\n[Main] Starting pipeline modules...\n")

    receiver.start()
    time.sleep(0.1)

    capture.start()
    time.sleep(0.1)

    aggregator.start()
    time.sleep(0.1)

    packetizer.start()
    time.sleep(0.1)

    sender.start()
    time.sleep(0.1)

    gen1_thread.start()
    gen2_thread.start()

    print("\n[Main] Pipeline running. Close the spectrum window or press Ctrl+C to stop.\n")

    try:
        # blocks until the spectrum window is closed
        run_spectrum_display(receiver)
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received")
    finally:
        print("\n[Main] Stopping pipeline...")
        sender.stop()
        packetizer.stop()
        aggregator.stop()
        capture.stop()
        receiver.stop()
        gen1.close()
        gen2.close()

        print("\n" + "=" * 60)
        print("  Pipeline stopped cleanly")
        print(f"  Generator 1  : {gen1._pkt_count} packets sent")
        print(f"  Generator 2  : {gen2._pkt_count} packets sent")
        print(f"  Aggregated   : {aggregator.chunks_emitted} chunks")
        print(f"  Transmitted  : {sender.packets_sent} packets")
        print(f"  Received     : {receiver.data_received} packets")
        print("=" * 60)


if __name__ == "__main__":
    main()