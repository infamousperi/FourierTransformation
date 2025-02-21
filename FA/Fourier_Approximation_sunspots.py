import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, ifft


# Function to load sunspot data from a file
def load_sunspot_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    years = data.values[:, 0::2].flatten()  # Extract years
    sunspot_numbers = data.values[:, 1::2].flatten()  # Extract sunspot numbers
    return years, sunspot_numbers


# Function to analyze sunspot data and find the dominant period
def analyze_sunspot_data(sunspot_numbers):
    # Calculate the power spectrum using FFT
    power_spectrum, xf, yf = calculate_power_spectrum(sunspot_numbers)

    # Exclude the zero frequency component to avoid division by zero
    xf_nonzero = xf[1:]
    power_spectrum_nonzero = power_spectrum[1:]

    # Find the dominant frequency and corresponding period
    dominant_frequency_nonzero = xf_nonzero[np.argmax(power_spectrum_nonzero)]
    dominant_period_nonzero = 1 / dominant_frequency_nonzero

    return xf, power_spectrum, dominant_period_nonzero, yf


# Function to calculate the dominant periods in the sunspot data
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


# Function to calculate the power spectrum of sunspot numbers using FFT
def calculate_power_spectrum(sunspot_numbers, T=1.0):
    N = len(sunspot_numbers)
    yf = fft(sunspot_numbers)
    power_spectrum = np.abs(yf[:N // 2]) ** 2
    xf = fftfreq(N, T)[:N // 2]
    return power_spectrum, xf, yf


# Function to modify the spectrum and reconstruct the signal
def modify_spectrum_and_reconstruct(yf, sunspot_numbers, k_greater_than=None, k_less_than=None):
    yf_mod = yf.copy()
    N = len(sunspot_numbers)
    freqs = fftfreq(N, 1. / N)

    # Remove largest components (k > specified threshold) if specified
    if k_greater_than is not None:
        yf_mod[np.abs(freqs) > k_greater_than] = 0

    # Remove smallest components (k < specified threshold) if specified
    if k_less_than is not None:
        yf_mod[np.abs(freqs) < k_less_than] = 0

    # Perform inverse FFT to reconstruct the signal
    reconstructed_signal = ifft(yf_mod)

    return reconstructed_signal, yf_mod


# Function to plot sunspot numbers over years
def show_sunspot_numbers_years(years, sunspot_numbers):
    plt.figure(figsize=(14, 6))
    plt.plot(years, sunspot_numbers, label='Sunspot Numbers')
    plt.title('Sunspot Numbers Over Years')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Numbers')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Function to plot the power spectrum of sunspot numbers
def show_sunspot_power_spectrum(xf, power_spectrum):
    plt.figure(figsize=(12, 6))

    # Exclude the zero frequency component to avoid plotting it
    xf_nonzero = xf[1:]
    power_spectrum_nonzero = power_spectrum[1:]

    plt.plot(xf_nonzero, power_spectrum_nonzero, label='Power Spectrum')
    plt.yscale('log')
    plt.title('Power Spectrum')
    plt.xlabel('Frequency [1/Year]')
    plt.ylabel('Power')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# Function to plot the original and reconstructed sunspot numbers
def show_sunspot_reconstructed_signal(years, sunspot_numbers, reconstructed_signal):
    plt.figure(figsize=(14, 6))
    plt.plot(years, sunspot_numbers, label='Original Sunspot Numbers')
    plt.plot(years, reconstructed_signal.real, label='Reconstructed Sunspot Numbers', linestyle='--')
    plt.title('Original and Reconstructed Sunspot Numbers')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Numbers')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
