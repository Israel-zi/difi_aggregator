# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

**Full pipeline (with live spectrum display):**
```bash
python src/main.py
```

**Individual modules (each has a `__main__` block for standalone testing):**
```bash
python src/core/difi_packet.py        # Packet encode/decode self-test
python src/modules/generator.py       # Send a test DIFI stream
python src/modules/input_capture.py   # Listen on UDP ports
python src/modules/aggregator.py      # Aggregation logic test
python src/modules/packetizer.py      # Packetizer test
python src/modules/receiver.py        # Receive + display FFT spectrum
```

There are no configured test runners, linters, or CI pipelines.

## Architecture

This is a **DIFI (VITA-49.2) signal aggregation proof-of-concept** in Python 3.12. It ingests multiple independent RF signal streams over UDP, aggregates and re-packetizes them into a single unified DIFI stream, and displays the real-time FFT spectrum.

**Pipeline (left to right):**
```
DifiGenerator (x2, ports 50001/50002)
  → InputCapture (multi-port UDP listener, one thread per port)
  → Aggregator (buffers per-stream samples, emits when all streams reach chunk_size)
  → Packetizer (concatenates streams into single DIFI payload, stream_id=0xAA000000)
  → DifiSender (UDP unicast to 127.0.0.1:50010)
  → DifiReceiver (listens on 50010, maintains rolling IQ buffer, renders spectrum)
```

All stages communicate via `queue.Queue` objects; each runs in its own thread. `src/main.py` wires them together and blocks on the spectrum display (Matplotlib animation or PyQtGraph).

**Default constants in `src/main.py`:**
- Sample rate: 48 kHz, 1024 samples/packet, 16-bit complex IQ, RF ref: 437 MHz
- Stream 1: stream_id=0x00000001, 2 kHz CW tone, port 50001
- Stream 2: stream_id=0x00000002, 6 kHz CW tone, port 50002

## Key Files

| Path | Role |
|------|------|
| `src/core/difi_packet.py` | VITA-49.2 packet builder & parser (`DifiDataPacket`, `DifiContextPacket`) |
| `src/modules/generator.py` | Generates CW/BW signals and sends them as DIFI packets over UDP |
| `src/modules/input_capture.py` | `InputCapture` + `PortListener` — multi-threaded UDP receiver and packet parser |
| `src/modules/aggregator.py` | Buffers per-stream samples; emits `AggregatedChunk` when all streams are ready |
| `src/modules/packetizer.py` | Merges aggregated chunks into a single DIFI packet stream |
| `src/modules/sender.py` | UDP transmission of (context_bytes, data_bytes) pairs |
| `src/modules/receiver.py` | UDP reception + rolling IQ buffer + FFT spectrum display |
| `src/ui/tx_gui.py` | PySide6 TX GUI with live PyQtGraph spectrum |
| `src/ui/config_gui.py` | PySide6 configuration GUI for generator parameters |

## DIFI Packet Format Notes

`DifiDataPacket` encodes 16-bit complex IQ samples (`int16` I, `int16` Q interleaved). The packet class/code field distinguishes data packets (0x0000) from context packets (0x0001). Timestamps use integer seconds + picosecond fractional encoding per IEEE-ISTO Std 4900-2021 v1.3.0. The parser in `difi_packet.py` auto-detects packet type from the header.
