import numpy as np
import matplotlib.pyplot as plt

from modules.generator import CwConfig, CwDifiGenerator
from modules.receiver import DifiReceiver


def plot_spectrum(signal: np.ndarray, fs_hz: float, title: str = "Spectrum"):
    N = len(signal)

    # Window to reduce leakage (optional but helps visuals)
    window = np.hanning(N).astype(np.float64)
    xw = signal * window

    X = np.fft.fft(xw)
    freqs = np.fft.fftfreq(N, d=1.0 / fs_hz)

    # Shift so 0 Hz is in the center
    Xs = np.fft.fftshift(X)
    fs = np.fft.fftshift(freqs)

    mag_db = 20 * np.log10(np.abs(Xs) + 1e-12)

    peak_freq = fs[np.argmax(mag_db)]
    peak_val = np.max(mag_db)

    plt.figure()
    plt.plot(fs, mag_db)
    plt.title(f"{title} (peak ~ {peak_freq:.1f} Hz)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    # zoom a bit around 0 (optional)
    # plt.xlim(-5000, 5000)

    plt.show()


def main():
    cfg = CwConfig(
        fs_hz=48000.0,
        tone_hz=2000.0,
        amplitude=0.8,
        samples_per_packet=1024,
        num_packets=5,
    )

    gen = CwDifiGenerator(cfg)
    packets = gen.generate_stream(stream_id=1)

    rx = DifiReceiver()
    signal = rx.reconstruct_signal(packets)

    plot_spectrum(signal, cfg.fs_hz, title="Reconstructed CW Spectrum")


if __name__ == "__main__":
    main()