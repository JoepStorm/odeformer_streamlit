import numpy as np
import sympy as sp

# Define the symbolic variable and time parameter
t_string = sp.Symbol('t')


def create_trajectory(input_strings, times):
    """
    Convert a string formula into a callable function that uses numpy

    Parameters:
    - formula_string: Array of string representations of the equation. One string per dimension.
    - times: The time series to evaluate the function at.

    Returns:
    - A function that takes a time value and returns the result
    """
    num_dimensions = len(input_strings)
    trajectory = np.zeros((times.shape[0], num_dimensions))
    for dim in range(num_dimensions):
        # Parse the string into a sympy expression
        expr = sp.sympify(input_strings[dim])

        # Convert to a numpy-compatible function
        func = sp.lambdify(t_string, expr, "numpy")

        trajectory[:, dim] = func(times)

    if input_strings[1] == "0":
        trajectory = trajectory[:,0].reshape(-1, 1)
    elif input_strings[2]  == "0":
        trajectory = trajectory[:,0:2]

    return trajectory


# # Example usage
# # formula = ["5 * cos(0.3 * t)", "5 * sin(0.3 * t)", "2 * exp(-0.1 * t)"]
# formula = ["5 * cos(0.3 * t)", "5 * sin(0.3 * t)", "0"]
# # formula = ["5 * cos(0.3 * t)", "0", "0"]
#
# # Now you can use it with different t values
# times = np.linspace(0, 10, 100)
#
# trajectory = create_trajectory(formula, times)
