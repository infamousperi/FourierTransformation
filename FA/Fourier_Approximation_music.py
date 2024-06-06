import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def plot_waveform(y, sr, title, start=0, end=None):
    plt.figure(figsize=(14, 5))
    if end:
        librosa.display.waveshow(y[start:end], sr=sr)
    else:
        librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def compute_amplitude_spectrum(y, sr):
    Y = np.fft.fft(y)
    Y_mag = np.abs(Y)
    freq = np.fft.fftfreq(len(Y), 1 / sr)
    return Y, Y_mag, freq


def plot_amplitude_spectrum(freq, Y_mag, sr):
    plt.figure(figsize=(14, 5))
    plt.plot(freq, Y_mag)
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sr / 2)  # Show only positive frequencies
    plt.show()


def split_signal(Y, freq, cutoff):
    Y_high = np.copy(Y)
    Y_low = np.copy(Y)

    Y_high[np.abs(freq) < cutoff] = 0  # Remove low frequencies
    Y_low[np.abs(freq) >= cutoff] = 0  # Remove high frequencies

    return Y_high, Y_low


def identify_top_frequencies(Y, freq, top_n=5):
    top_indices = np.argsort(np.abs(Y))[-top_n:]
    top_freqs = freq[top_indices]
    return top_freqs


def reconstruct_signal(Y):
    return np.fft.ifft(Y).real


def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


def save_audio(directory, filename, data, sr):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    sf.write(filepath, data, sr)
    print(f"File saved: {filepath}")
