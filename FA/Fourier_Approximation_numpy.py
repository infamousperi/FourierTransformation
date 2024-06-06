import numpy as np
import matplotlib.pyplot as plt


def compute_fourier_component(x_values, y_values, k_index):
    # Calculate period length
    period_length = x_values[-1] - x_values[0]

    # Calculate the frequency
    angular_frequency = 2 * np.pi * k_index / period_length

    # Calculate the Fourier components using the trapezoidal rule
    a_k = 2 / period_length * np.trapz(y_values * np.cos(angular_frequency * x_values), x_values)
    b_k = 2 / period_length * np.trapz(y_values * np.sin(angular_frequency * x_values), x_values)

    # Return the complex Fourier component
    return a_k - 1j * b_k


def fourier_approximation(x_values, y_values, m_value):
    # Calculate the Fourier coefficients for harmonic_index = -m_value to m_value
    coefficients = [compute_fourier_component(x_values, y_values, k) for k in range(-m_value, m_value + 1)]

    # Approximate function
    approximation = np.zeros_like(x_values, dtype=np.complex128)
    period_length = x_values[-1] - x_values[0]

    for k, c_k in enumerate(coefficients):
        k_index = k - m_value
        approximation += c_k * np.exp(1j * k_index * 2 * np.pi * x_values / period_length)

    return approximation.real


# Define your functions
def f1(x):
    return 1 / (np.exp(x) + np.exp(-x))


def f2(x):
    return np.exp(x)


def f3(x):
    return np.exp(np.abs(x))


def plot_fourier_approximations(x_values, y_values, m_values, function_name='f(x)'):
    # Plot the original function
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, label=f'Original function {function_name}', linewidth=2)

    # Plot the Fourier approximations for different values of m
    for m in m_values:
        fapp = fourier_approximation(x_values, y_values, m)
        plt.plot(x_values, fapp, label=f'Fourier approximation (2m + 1 = {2 * m + 1})')

    # Add plot details
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Fourier Approximation with Different Numbers of Components')
    plt.legend()
    plt.grid(True)
    plt.show()


def find_fourier_components_for_error_tolerance(x_values, y_values, tolerance=0.001, function_name='f(x)',
                                                max_m_value=100):
    m_value = 1
    previous_error = float('inf')

    while m_value <= max_m_value:
        approximation = fourier_approximation(x_values, y_values, m_value)

        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.nanmax(np.abs((approximation - y_values) / y_values)) * 100
            print(f'M_Value: {m_value}, Relative Error: {relative_error}%')

        if relative_error <= tolerance:
            break

        if relative_error > previous_error and m_value > 1:
            print(f"Relative error increased at m = {m_value}. Stopping early.")
            m_value -= 1
            break

        previous_error = relative_error
        m_value += 1

    # Plot the original and approximated functions
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, label=f'Original function {function_name}', linewidth=2)
    plt.plot(x_values, approximation, label=f'Fourier approximation (2m + 1 = {2 * m_value + 1})', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Fourier Approximation with m = {m_value} (2m + 1 = {2 * m_value + 1} components)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return m_value
