import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import scipy.fft as fft


# Function to load an audio file
def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


# Function to plot the waveform of an audio signal
def plot_waveform(y, sr, title, start=0, end=None):
    plt.figure(figsize=(14, 5))
    if end:
        librosa.display.waveshow(y[start:end], sr=sr)
    else:
        librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


# Function to compute the amplitude spectrum of an audio signal
def compute_amplitude_spectrum(y, sr):
    Y = fft.fft(y)
    Y_mag = np.abs(Y)
    freq = fft.fftfreq(len(Y), 1 / sr)
    return Y, Y_mag, freq


# Function to plot the amplitude spectrum of an audio signal
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


# Function to split the signal into high and low frequency components
def split_signal(Y, freq, cutoff):
    Y_high = np.copy(Y)
    Y_low = np.copy(Y)

    Y_high[np.abs(freq) < cutoff] = 0  # Remove low frequencies
    Y_low[np.abs(freq) >= cutoff] = 0  # Remove high frequencies

    return Y_high, Y_low


# Function to identify the top N frequencies in the signal
def identify_top_frequencies(Y, freq, top_n=5):
    # Only consider positive frequencies
    positive_indices = np.where(freq > 0)
    Y_positive = Y[positive_indices]
    freq_positive = freq[positive_indices]

    # Find indices of the top N frequencies
    top_indices = np.argsort(np.abs(Y_positive))[-top_n:]
    top_freqs = freq_positive[top_indices]

    return top_freqs


# Function to reconstruct the signal from its frequency components
def reconstruct_signal(Y):
    return fft.ifft(Y).real


# Function to calculate the mean squared error between the original and reconstructed signals
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


# Function to save an audio file
def save_audio(directory, filename, data, sr):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    sf.write(filepath, data, sr)
    print(f"File saved: {filepath}")


# Function to plot the original and separated waveforms (high and low frequencies)
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
    plt.tight_layout()
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
    plt.tight_layout()
    plt.show()
