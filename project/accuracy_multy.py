
'''
Project BRIGHENTI

This script is used to obtain the correlation between space-step
and the error of the numerical solution of the 1D steady-state.

multi processing version
'''

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import concurrent.futures

# Constants
Pe = 15   # Assigned
ks = '1'  # '0' for CD scheme, '1' for Upwind scheme
L = 1

def compute(dx):
    M = int(L / dx) + 1  # Compute M from dx
    if M < 3:
        return None  # Cannot compute for M less than 3
    dx = L / (M - 1)  # Recalculate dx to ensure consistency
    x = np.linspace(start=0, stop=L, num=M)

    # Grid Peclet number
    Pec = Pe * dx

    # Empty boundary array
    T = np.zeros(M)
    # Boundary conditions
    T[0] = 0
    T[-1] = 1

    ## Weight and matrix calculation A
    if ks == '0':
        ww = -(0.5 * Pec + 1)
        wc = 2
        we = 0.5 * Pec -1
    elif ks == '1':
        ww = - Pec -1
        wc = 2 + Pec
        we = -1
    else:
        raise ValueError("Invalid value for ks. Use '0' for CD scheme or '1' for Upwind scheme.")

    diagl = ww * np.ones((M - 2))
    diagp = wc * np.ones((M - 2))
    diagu = we * np.ones((M - 2))

    A = sps.spdiags([diagl, diagp, diagu], [-1, 0, 1], M - 2, M - 2, format='csc')

    ## Resolution
    # Known term
    rhs = np.zeros(M - 2)

    rhs[0] = rhs[0] - ww * T[0]
    rhs[-1] = rhs[-1] - we * T[-1]

    Tint = sps.linalg.spsolve(A, rhs)

    def solution(T, Tint, M, withArrays=True):
        if withArrays:
            T[1:M -1] = Tint[0:M -2]
        else:
            for jx in range(1, M -1):
                T[jx] = Tint[jx -1]
        return T

    Texact = (np.exp(Pe * x) -1) / (np.exp(Pe) -1)
    T = solution(T, Tint, M, withArrays=True)

    # Compute errors
    error_L2 = np.sqrt(np.mean((T - Texact) ** 2))
    error_Linf = np.max(np.abs(T - Texact))

    # Return dx and errors
    return dx, error_L2, error_Linf

if __name__ == '__main__':
    # Define delta x values logarithmically spaced
    min_dx = 1e-5  # Adjust the minimum delta x value as needed
    max_dx = 1e-1  # Adjust the maximum delta x value as needed
    num_points = 50  # Number of points in the delta x array
    delta_x_values = np.logspace(np.log10(min_dx), np.log10(max_dx), num=num_points)

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(compute, delta_x_values))

    # Remove any None results (if M < 3)
    results = [res for res in results if res is not None]

    # Extract results
    delta_x_list = [result[0] for result in results]
    error_L2_list = [result[1] for result in results]
    error_Linf_list = [result[2] for result in results]

    # Plotting the errors
    plt.figure()
    plt.plot(delta_x_list, error_Linf_list, label='$L_\infty$ Error', marker='x')
    plt.plot(delta_x_list, error_L2_list, label='$L_2$ Error', marker='o')

    plt.xlabel('$\Delta x$')
    plt.ylabel('Error')

    plt.xscale('log')
    plt.yscale('log')

    plt.legend()
    plt.title('Error vs. grid spacing')
    plt.grid(True, which="minor", ls="--", axis='x', linewidth=0.3)
    plt.grid(True, which="major", ls="--", axis='y', linewidth=0.3)

    plt.show()