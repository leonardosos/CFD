'''
Project BRIGHENTI

This script is used to obatain the correlation between the Peclet number
and the convergence of the numerical solution of the 1D steady-state.

It plot the solution for a specific M and Peclet number using 
- the `plot_solution_steady` function
- Matplotlib custom plotting functions

'''

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from plot import plot_solution_steady

## Manual Parameter input
'''
M = int(input('Insert number of points > '))
Pe = float(input('Insert Peclet number > '))
ks = input('0 - CD, 1 - Upwind > ')
L=1
'''

Pe = 15   # Assigned
ks = '1'  # 0 CD, 1 Upwind
L = 1
M = 100

fig, ax = plt.subplots()

# Compute and Plot the Exact solution
x = np.linspace(start=0, stop=1, num=100)
Texact = (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
ax.plot(x, Texact, lw=1, color='black', linestyle='dashed', label = '$T_{exact}$')

## Grid setup
dx = L/(M-1)  # space-step
x = np.linspace(start=0, stop=L, num=M)

# grid Peclet number
Pec = Pe*dx

# Empty boundary array
T = np.zeros(M)
# Boundary conditions
T[0] = 0
T[-1] = 1

## Weight and matrix calculation A 
match ks:
    case '0':
        ww = -(0.5*Pec + 1)
        wc = 2
        we = 0.5*Pec - 1
    case '1':
        ww = - Pec - 1
        wc = 2 + Pec
        we = -1

diagl = ww*np.ones((M-2))
diagp = wc*np.ones((M-2))
diagu = we*np.ones((M-2))

A = sps.spdiags([diagl,diagp,diagu],[-1,0,1], M-2, M-2, format='csc')

## Resolution
#known term
rhs= np.zeros(M-2)

rhs[0] = rhs[0] - ww*T[0]
rhs[-1] = rhs[-1] - we*T[-1]

Tint = sps.linalg.spsolve(A,rhs)

def solution(T,Tint,M,withArrays=True):
    if withArrays:
        T[1:M-1] = Tint[0:M-2]
    else:
        for jx in range(1,M-1):
            T[jx] = Tint[jx-1]
    return T

T = solution(T,Tint,M,withArrays=True)

marker_style = 'x' if Pec > 2 else '.'
ax.plot(x, T, marker=marker_style, label=f'M = {M}, Pe = {round(Pec, 2)}')

plot_solution_steady(x, T, Texact, Pe, dx)

plt.legend()
plt.show()
