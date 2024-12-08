# Crank-Nicolson scheme diffusion

import numpy as np
import os
from plot import plot_animation
import scipy.sparse as sps # sparse matrix library

def ftcs_scheme(T,M,nt,Fo):
    for jt in range(1,nt): # 
        for jx in range(1,M-1): # only internal poitns
            T[jx,jt] = Fo*(T[jx-1,jt-1] + T[jx+1,jt-1]) + \
                (1-2*Fo)*T[jx,jt-1]
            
        T[0,jt] = 1
        T[-1,jt] = 1
    return T 

# ADD THIS FUNCTION FROM FTCS.PY
def crank_nicolson_scheme(T,M,nt,Fo):
    # main diagonal 
    diagn = (1+Fo)*np.ones(M-2)

    # off diagonal, upper and lower diagonal
    off_diagn = -0.5*Fo*np.ones(M-2)

    # sistem matrix
    A = sps.spdiags([off_diagn, diagn, off_diagn],[-1,0,1], M-2, M-2, format='csc')
    
    rhs = np.zeros(M-2) # right hand side

    # solution loop
    for jt in range(1,nt):
        for jx in range(1,M-1):
            rhs[jx-1] = 0.5*Fo*(T[jx-1,jt-1] + T[jx+1,jt-1]) + (1-Fo)*T[jx,jt-1]
        rhs[0] += 0.5*Fo*T[0,jt-1]
        rhs[-1] += 0.5*Fo*T[-1,jt-1]
        # solve for T at internal points
        Tint = sps.linalg.spsolve(A,rhs)
        T[1:M-1,jt] = Tint
        # update boundary conditions
        T[0,jt] = 1
        T[-1,jt] = 1
    
    return T

#input parameters
'''
dt = float(input("Enter time step: "))
M = int(input("Enter num of grid points: "))
tf = float(input("Enter final time: "))
alpha = float(input("Enter thermal diffusivity: "))
L = 1
'''

"""
Enter time step: 0.001
Enter num of grid points: 11
Enter final time: 1
Enter thermal diffusivity: 0.01
"""

dt = 0.1
M = 11
tf = 1
alpha = 0.01
L = 1

# Fourier number=10 , ok stability

#setup time array and grid
nt = round(tf/dt) +1 # total number of time steps
t = np.linspace(start=0, stop=tf, num=nt) # time array
dx = L/(M-1) # grid spacing
# if the have M grid points, we have M-1 intervals
x = np.linspace(start=0, stop=L, num=M) # X array of grid points

# Grid fourier number calculation
Fo = alpha*dt/dx**2

# Initial T field
T = np.zeros((len(x),len(t))) # matrix to store T field (discetized) at all time steps

# Boundary conditions of the rod
T[0,0] = 1   # First element
T[-1,0] = 1  # Last element

# exact T calculation
Texact = np.zeros_like(T) # array to store exact T field 
Texact[:,:] = T[:,:] # copy the value and avoid referencing

# compute the T exact solution
for jt in range(1,nt):  #not in 0 time because we already have the initial condition
    for jx in range(0,len(x)):
        summ = 0
        for jm in range(1,101): # 100 terms of the fourier series
            summ += 4*np.sin((2*jm-1)*x[jx]*np.pi)* \
                np.exp(-alpha*(2*jm-1)**2*np.pi**2*t[jt])/ \
                ((2*jm-1)*np.pi)
        Texact[jx,jt] = 1 - summ

#T = ftcs_scheme(T,M,nt,Fo)
T = crank_nicolson_scheme(T,M,nt,Fo)


plot_animation(x,T,Texact,nt,t,dx)
           
