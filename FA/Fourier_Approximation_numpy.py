import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# Function to compute a single Fourier component for a given k
def compute_fourier_component(x_values: np.array, y_values: np.array, k: int) -> complex:
    TWO_PI = 2 * np.pi
    step_size = x_values[1] - x_values[0]
    # Compute the integrand for the Fourier component
    integrand = y_values * (np.cos(x_values * k) - 1j * np.sin(x_values * k))
    # Return the computed Fourier component using the trapezoidal rule for integration
    return np.trapz(integrand, dx=step_size) / TWO_PI


# Function to compute the Fourier approximation of a given function
def fourier_approximation(f: Callable[[float], float], a: float, N: int, m: int) -> Tuple[
    np.array, np.array, np.array, np.array]:
    x_values = np.linspace(-a, a, N)
    y_values = f(x_values)
    x_rescaled = x_values * np.pi / a
    coefficients = np.zeros(2 * m + 1, dtype=complex)

    # Compute the zeroth Fourier component (average value)
    f0 = compute_fourier_component(x_rescaled, y_values, 0)
    coefficients[m] = f0
    approximation = f0 * np.ones(x_values.shape)

    # Compute the remaining Fourier components and build the approximation
    for k in range(1, m + 1):
        fk = compute_fourier_component(x_rescaled, y_values, k)
        fnegk = compute_fourier_component(x_rescaled, y_values, -k)
        coefficients[m + k] = fk
        coefficients[m - k] = fnegk
        # Add the k-th and -k-th components to the approximation
        approximation += (fk + fnegk) * np.cos(x_rescaled * k) + 1j * (fk - fnegk) * np.sin(x_rescaled * k)

    return approximation.real


# Function to plot the original function and its Fourier approximations for various m values
def plot_fourier_approximations(x_values, y_values, m_values, function_name='f(x)'):
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, label=f'Original function {function_name}', linewidth=2)

    # Plot the Fourier approximations for different values of m
    for m in m_values:
        fapp = fourier_approximation(lambda x: np.interp(x, x_values, y_values), x_values[-1], len(x_values), m)
        plt.plot(x_values, fapp, label=f'Fourier approximation (2m + 1 = {2 * m + 1})')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Fourier Approximation with Different Numbers of Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Function to plot the error of the Fourier approximation and the approximation itself
def plot_error_and_approximation(x_values, y_values, yappr, ms, errors, m, label=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot the error vs. m
    ax1.plot(ms, errors)
    ax1.set_ylabel('Error (%)')
    ax1.set_xlabel('m')
    ax1.set_yscale("log")
    ax1.set_title(f'Error of Fourier Approximation for {label}')
    ax1.grid(True)

    # Plot the original function and its approximation
    ax2.plot(x_values, y_values, label='Original function')
    ax2.plot(x_values, yappr, label=f'Approximation (m={m})', linestyle="-.")
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title(f'Fourier Approximation for {label}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Function to find the optimal number of Fourier components to achieve a desired error
def find_optimal_fourier_components(f: Callable[[float], float], x_values, desired_error=0.1, m_step=1):
    m = 0
    y_values = f(x_values)
    yappr = fourier_approximation(f, x_values[-1], len(x_values), m)
    err = np.nanmax(np.abs((yappr - y_values) / y_values)) * 100
    ms = [m]
    errors = [err]
    err0 = err
    print(f"m={m}:\tError = {err:.4f}%")

    # Increment m until the desired error is achieved
    while err > desired_error:
        m += m_step
        yappr = fourier_approximation(f, x_values[-1], len(x_values), m)
        err = np.nanmax(np.abs((yappr - y_values) / y_values)) * 100
        ms.append(m)
        errors.append(err)
        print(f"m={m}:\tError = {err:.4f}%")
        if err0 < err:
            break

    return x_values, y_values, yappr, ms, errors, m


# Example functions to approximate using Fourier series
def f1(x):
    return 1 / (np.exp(x) + np.exp(-x))


def f2(x):
    return np.exp(x)


def f3(x):
    return np.exp(np.abs(x))
