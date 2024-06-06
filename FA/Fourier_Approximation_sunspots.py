import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, ifft


def load_sunspot_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    years = data.values[:, 0::2].flatten()
    sunspot_numbers = data.values[:, 1::2].flatten()
    return years, sunspot_numbers


def analyze_sunspot_data(years, sunspot_numbers):
    # Calculate the power spectrum using FFT
    power_spectrum, xf, yf = calculate_power_spectrum(sunspot_numbers)

    # Exclude the zero frequency component to avoid division by zero
    xf_nonzero = xf[1:]
    power_spectrum_nonzero = power_spectrum[1:]

    # Find the dominant frequency and corresponding period
    dominant_frequency_nonzero = xf_nonzero[np.argmax(power_spectrum_nonzero)]
    dominant_period_nonzero = 1 / dominant_frequency_nonzero

    return xf, power_spectrum, dominant_period_nonzero, yf


def calculate_dominant_periods(sunspot_numbers, num_periods=5):
    # Calculate the power spectrum using FFT
    power_spectrum, xf, yf = calculate_power_spectrum(sunspot_numbers)

    # Exclude the zero frequency component to avoid division by zero
    xf_nonzero = xf[1:]
    power_spectrum_nonzero = power_spectrum[1:]

    # Find the dominant frequencies and corresponding periods
    dominant_indices = np.argsort(power_spectrum_nonzero)[-num_periods:][::-1]
    dominant_frequencies = xf_nonzero[dominant_indices]
    dominant_periods = 1 / dominant_frequencies

    for i, period in enumerate(dominant_periods, 1):
        print(f"Dominant period {i}: {period:.2f} years")

    return dominant_periods


def calculate_power_spectrum(sunspot_numbers, yf_mod=None):
    # Calculate the power spectrum using FFT
    N = len(sunspot_numbers)
    T = 1.0  # Time interval in years
    if yf_mod is None:
        yf = fft(sunspot_numbers)
    else:
        yf = yf_mod
    xf = fftfreq(N, T)[:N // 2]

    # Compute the power spectrum
    power_spectrum = 2.0 / N * np.abs(yf[:N // 2])
    return power_spectrum, xf, yf


def modify_spectrum_and_reconstruct(yf, k_greater_than=None, k_less_than=None):
    # Calculate the power spectrum
    power_spectrum = np.abs(yf)

    # Remove largest components (k > 20) if specified
    yf_mod = yf.copy()
    if k_greater_than is not None:
        indices_greater = np.where(power_spectrum > k_greater_than)[0]
        yf_mod[indices_greater] = 0

    # Remove smallest components (k < 5) if specified
    if k_less_than is not None:
        indices_less = np.where(power_spectrum < k_less_than)[0]
        yf_mod[indices_less] = 0

    # Perform inverse FFT to reconstruct the signal
    reconstructed_signal = ifft(yf_mod)

    return reconstructed_signal, yf_mod


def show_sunspot_numbers_years(years, sunspot_numbers):
    plt.figure(figsize=(14, 6))
    plt.plot(years, sunspot_numbers, label='Sunspot Numbers')
    plt.title('Sunspot Numbers Over Years')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Numbers')
    plt.legend()
    plt.grid()
    plt.show()


def show_sunspot_power_spectrum(xf, power_spectrum):
    plt.figure(figsize=(12, 6))
    plt.plot(xf, power_spectrum)
    plt.title('Power Spectrum of Sunspot Numbers')
    plt.xlabel('Frequency (1/year)')
    plt.ylabel('Power')
    plt.grid()
    plt.show()


def show_sunspot_reconstructed_signal(years, sunspot_numbers, reconstructed_signal):
    plt.figure(figsize=(14, 6))
    plt.plot(years, sunspot_numbers, label='Original Sunspot Numbers')
    plt.plot(years, reconstructed_signal.real, label='Reconstructed Sunspot Numbers', linestyle='--')
    plt.title('Original and Reconstructed Sunspot Numbers')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Numbers')
    plt.legend()
    plt.grid()
    plt.show()
