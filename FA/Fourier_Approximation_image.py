import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


# Function to load an image and convert it to grayscale
def load_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.array(img)


# Function to perform 2D FFT on an image and compute the magnitude spectrum
def perform_fft(image_array):
    fft_img = fft2(image_array)
    fft_img_shifted = fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_img_shifted)
    return fft_img_shifted, magnitude_spectrum


# Function to plot the original image and its magnitude spectrum
def plot_magnitude_spectrum(original_image, magnitude_spectrum, title):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Set white background for each subplot
    for ax in axes:
        ax.set_facecolor('white')

    # Display original image
    axes[0].imshow(original_image, cmap='inferno')
    axes[0].set_title('Original Image')

    # Display magnitude spectrum
    axes[1].imshow(np.log1p(magnitude_spectrum), cmap='inferno')
    axes[1].set_title(title)

    plt.tight_layout()
    plt.show()


# Function to filter FFT components based on a list of thresholds
def filter_fft_components(fft_img_shifted, magnitude_spectrum, thresholds):
    max_amplitude = np.max(magnitude_spectrum)
    filtered_images = []
    for threshold in thresholds:
        threshold_value = (threshold / 100) * max_amplitude
        filtered_fft = np.where(magnitude_spectrum > threshold_value, fft_img_shifted, 0)
        filtered_img = np.abs(ifft2(ifftshift(filtered_fft)))
        filtered_images.append((threshold, filtered_img))
    return filtered_images


# Function to plot filtered images and their corresponding magnitude spectrums
def plot_filtered_images(original_image, filtered_images):
    for threshold, filtered_img in filtered_images:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Calculate and display FFT magnitude spectrum of the filtered image
        _, filtered_magnitude_spectrum = perform_fft(filtered_img)
        axes[0].imshow(np.log(1 + filtered_magnitude_spectrum), cmap='inferno')
        axes[0].set_title(f'Filtered Fourier Spectrum, {threshold:.5f}% filtered')

        # Display filtered image
        axes[1].imshow(filtered_img, cmap='inferno')
        axes[1].set_title(f'Reconstructed Image (Threshold: {threshold:.5f}%)')

        plt.tight_layout()
        plt.show()


# Function to compute the Mean Squared Error (MSE) between the original and reconstructed images
def compute_mse(original_image, reconstructed_image):
    return np.mean((original_image - reconstructed_image) ** 2)


# Function to find the maximum compression threshold that keeps the error within a desired percentage
def find_max_compression_threshold(original_image, fft_img_shifted, magnitude_spectrum, max_error_percentage):
    # Find the maximum value in the magnitude spectrum
    max_amplitude = np.max(magnitude_spectrum)

    # Initialize the low and high bounds for binary search
    low, high = 0, 100
    best_threshold = 0

    # Perform binary search to find the maximum threshold within error limit
    while low <= high:
        # Calculate the midpoint of the current range
        mid = (low + high) / 2

        # Calculate the threshold value corresponding to the midpoint percentage
        threshold_value = (mid / 100) * max_amplitude

        # Apply the threshold to filter the FFT image
        filtered_fft = np.where(magnitude_spectrum > threshold_value, fft_img_shifted, 0)

        # Compute the inverse FFT to get the compressed image
        filtered_img = np.abs(ifft2(ifftshift(filtered_fft)))

        # Calculate the Mean Squared Error (MSE) between the original and compressed images
        mse = compute_mse(original_image, filtered_img)

        # Calculate the error percentage
        error_percentage = (mse / np.mean(original_image ** 2)) * 100

        # Log the current state of the binary search
        print(f'Low: {low:.5f}%, High: {high:.5f}%, Mid: {mid:.5f}%, '
              f'Threshold Value: {threshold_value:.5f}, Error: {error_percentage:.5f}%')

        # Adjust the search range based on the error percentage
        if error_percentage <= max_error_percentage:
            # If the error is within the acceptable limit, update the best threshold
            best_threshold = mid
            low = mid + 0.00001  # Narrow the search upwards
        else:
            # If the error exceeds the limit, narrow the search downwards
            high = mid - 0.00001

    # Return the best threshold found within the error limit
    return best_threshold
