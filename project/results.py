'''
Project BRIGHENTI

This script is used to obatain the correlation between the Peclet number
and the convergence of the numerical solution of the 1D steady-state.
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
#M = 100

fig, ax = plt.subplots()

# Compute and Plot the Exact solution
x = np.linspace(start=0, stop=1, num=100)
Texact = (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
ax.plot(x, Texact, lw=1, color='black', linestyle='dashed', label = '$T_{exact}$')

# Compute the list of different M values for given grid Pèclet number
Pec_set_list = np.array((0.5, 1, 2, 3))  # Peclet number set
M_set_list = list()
for Pec_set in Pec_set_list:
    M_set = int(1+15/Pec_set)
    M_set_list.append(M_set)

# Compute and Plot the numerical solution
for M in M_set_list:

    ## Grid setup
    dx = L/(M-1); # space-step
    x = np.linspace(start=0, stop=L, num=M)

    #grid Peclet number
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

    T=solution(T,Tint,M,withArrays=True)

    if Pec > 2:
        ax.plot(x, T, marker='x' ,label = f'M = {M}, Pe = {round(Pec, 2)}')
    else:
        ax.plot(x, T, marker='.' ,label = f'M = {M}, Pe = {round(Pec, 2)}')

plt.legend()
plt.show()

#plot_solution_steady(x,T,Texact,Pe,dx)