# Advecttion

import numpy as np
import os
from plot import plot_animation
import scipy.sparse as sps # sparse matrix library

def ftcs_scheme(T,M,nt,Co):
    for jt in range(1,nt): # 
        for jx in range(1,M-1): # only internal poitns
            T[jx,jt] = T[jx,jt-1] + 0.5*Co*(T[jx+1,jt-1] - T[jx-1,jt-1])

    return T 

def upwind_scheme(T,M,nt,Co):
    for jt in range(1,nt): # 
        for jx in range(1,M-1): # only internal poitns
            T[jx,jt] = (1-Co)*T[jx,jt-1] + Co*T[jx-1,jt-1]

    return T 

def crank_nicolson_scheme(T,M,nt,Co):

    # upper digonal
    diagu = 0.25*Co*np.ones(M-2)
    # main diagonal 
    diagm = np.ones(M-2)
    # lower diagonal
    diagl = -0.25*Co*np.ones(M-2)

    # sistem matrix
    A = sps.spdiags([diagl,diagm,diagu],[-1,0,1], M-2, M-2, format='csc')
    
    rhs = np.zeros(M-2) # right hand side

    # solution loop
    for jt in range(1,nt):
        for jx in range(1,M-1):
            rhs[jx-1] = 0.25*Co*(T[jx-1,jt-1] - T[jx+1,jt-1]) + T[jx,jt-1]
        rhs[0] += 0.25*Co*T[0,jt-1]
        rhs[-1] -= 0.25*Co*T[-1,jt-1]
        # solve for T at internal points
        Tint = sps.linalg.spsolve(A,rhs)
        T[1:M-1,jt] = Tint
    
    return T


#if input parameters
'''
dt = float(input("Enter time step: "))
M = int(input("Enter num of grid points: "))
tf = float(input("Enter final time: "))
u = float(input("Enter way propagation speed: "))
L = 1
'''

# Courant number << 1, 
# numerical diffusion :() with upwind scheme (stable), worst case scenario
dt = 0.01
M = 101
tf = 2
u = 0.1
L = 1

# Courant number < 1, 
# numerical diffusion :() with upwind scheme (stable)
dt = 0.01
M = 101
tf = 2
u = 0.5
L = 1
'''

# Courant number = 1, 
# upwind scheme (stable) replacating the exact solution
dt = 0.01
M = 101
tf = 2
u = 1
L = 1

# Courant number > 1, 
# upwind scheme unstable, the wave traverl faster than one grid point per time step
dt = 0.01
M = 101
tf = 0.9
u = 1.2
L = 1
'''

# setup time array and grid
nt = round(tf/dt) +1 # total number of time steps
t = np.linspace(start=0, stop=tf, num=nt) # time array
dx = L/(M-1) # grid spacing
# if the have M grid points, we have M-1 intervals
x = np.linspace(start=0, stop=L, num=M) # X array of grid points

# Grid Courand number calculation
Co = u*dt/dx

# Initial T field
T = np.zeros((len(x),len(t))) # matrix to store T field (discetized) at all time steps

# Boundary conditions of the rod, already is 0
#T[0,0] = 0   # First element
#T[-1,0] = 0  # Last element

# Construct a logical array to store the location of the initial condition, the sin wave
j0 = np.where((x <= 0.1) & (x >= 0.0), True, False)
T[j0,0] = np.sin(10*np.pi*x[j0])

# plot T the initial condition
#plot_animation(x,T,T,1,t,dx)

# exact T calculation
Texact = np.zeros_like(T) # array to store exact T field 

# compute the T exact solution
for jt in range(1,nt):  #not in 0 time because we already have the initial condition
    xt = u*t[jt] # displacemnt at time t[jt]
    j0 = np.where(((x - xt) <= 0.1) & ((x - xt) >= 0), True, False) 
    Texact[j0,jt] = np.sin(10*np.pi*(x[j0] - xt))

# plot T with just moving signal
#plot_animation(x,Texact,Texact,80,t,dx)


'''
T = ftcs_scheme(T,M,nt,Co)  # Unstable advective scheme

print("Courant number: ", Co)
if Co >= 1:
    print("Courant number is greater than 1, the scheme is unstable")
# Always get the unstable scheme even different values of Courant number 
'''

#T = upwind_scheme(T,M,nt,Co)  # Stable advective scheme

T = crank_nicolson_scheme(T,M,nt,Co)  # Unconditional Stable advective scheme

plot_animation(x,T,Texact,nt,t,dx)
           
