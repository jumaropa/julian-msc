import numpy as np
import matplotlib.pyplot as plt
import os

def get_arrays(appr, elt, order, var, base_path="data/"):
    """
    Reads two arrays from a text file based on input parameters.
    Assumes filenames follow the format: "appr_elt_order_var.txt".
    Each file contains two lines, each representing an array.
    """
    filename = f"{appr}_{elt}_{order}_{var}.txt"
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        return [], []  # Return empty lists if the file does not exist

    with open(filepath, "r") as file:
        lines = file.readlines()

    if len(lines) < 2:
        return [], []  # Ensure there are at least two lines

    array1 = list(map(float, lines[0].strip().split()))
    array2 = list(map(float, lines[1].strip().split()))

    return np.array(array1), np.array(array2)

approach, element, order, variable = "NSF", "rav", 3, "p"
print(f"{approach} - {element} - order {order} - {variable}")

errors_L2, errors_Linf = get_arrays(approach, element, order, variable)

h_values = np.array([1.0, 0.5, 0.3333333333333333, 0.25])

# Perform Richardson extrapolation for L2 and Linf errors
print("\nRichardson Extrapolation Results:")
def richardson_extrapolation(h_vals, err_vals):
    estimates = []
    for i in range(len(h_vals) - 1):
        h1, h2 = h_vals[i:i+2]
        e1, e2 = err_vals[i:i+2]
        
        # Estimate order k
        k = np.log(e1 / e2) / np.log(h1 / h2)

        estimates.append(k)
    return estimates

richardson_L2 = richardson_extrapolation(h_values, errors_L2)
richardson_Linf = richardson_extrapolation(h_values, errors_Linf)

print("L2: ", richardson_L2)
print("Linf: ", richardson_Linf)

# Plot errors vs. grid spacing (log-log)
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_L2, 'o-', label="L2 Error")
plt.loglog(h_values, errors_Linf, 'o-', label="Linf Error")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title(f"{approach} - {element} - order {order} - {variable}")
plt.show()
