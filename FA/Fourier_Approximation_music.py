import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import scipy.fft as fft


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
    Y = fft.fft(y)
    Y_mag = np.abs(Y)
    freq = fft.fftfreq(len(Y), 1 / sr)
    return Y, Y_mag, freq


def plot_amplitude_spectrum(freq, Y_mag, sr, zoom_freq=None):
    plt.figure(figsize=(14, 10))

    # Plot the full amplitude spectrum
    plt.subplot(2, 1, 1)
    plt.plot(freq, Y_mag)
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Plot the zoomed-in amplitude spectrum
    if zoom_freq:
        zoom_freq_index = np.where(freq <= zoom_freq)[0][-1]
        plt.subplot(2, 1, 2)
        plt.plot(freq[:zoom_freq_index], Y_mag[:zoom_freq_index])
        plt.title('Zoomed Amplitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, zoom_freq)

    plt.tight_layout()
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
    return fft.ifft(Y).real


def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


def save_audio(directory, filename, data, sr):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    sf.write(filepath, data, sr)
    print(f"File saved: {filepath}")


def plot_separate_waveforms(y, y_high, y_low, sr, title, start=0, end=None):
    # Plot original and high frequencies
    plt.figure(figsize=(14, 5))
    if end:
        librosa.display.waveshow(y[start:end], sr=sr, alpha=0.5, label='Original')
        librosa.display.waveshow(y_high[start:end], sr=sr, color='r', alpha=0.5, label='High Frequencies')
    else:
        librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Original')
        librosa.display.waveshow(y_high, sr=sr, color='r', alpha=0.5, label='High Frequencies')
    plt.title(title + ' - High Frequencies')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Plot original and low frequencies
    plt.figure(figsize=(14, 5))
    if end:
        librosa.display.waveshow(y[start:end], sr=sr, alpha=0.5, label='Original')
        librosa.display.waveshow(y_low[start:end], sr=sr, color='g', alpha=0.5, label='Low Frequencies')
    else:
        librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Original')
        librosa.display.waveshow(y_low, sr=sr, color='g', alpha=0.5, label='Low Frequencies')
    plt.title(title + ' - Low Frequencies')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()