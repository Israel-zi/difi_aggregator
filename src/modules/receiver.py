import numpy as np


class DifiReceiver:
    def reconstruct_signal(self, packets):
        """
        Concatenate payloads in sequence order
        """
        packets_sorted = sorted(packets, key=lambda p: p.sequence)
        signal = np.concatenate([p.payload for p in packets_sorted])
        return signal

    def spectrum(self, signal, fs_hz):
        """
        Compute magnitude spectrum
        """
        N = len(signal)
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, d=1.0/fs_hz)
        return freqs, np.abs(spectrum)