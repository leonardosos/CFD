#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:09:14 2023

@author: ludovico
"""

# import warnings to hide a plot initialization warning (not necessary)
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# import numpy to work with arrays (https://numpy.org/)
import numpy as np

# import matplotlib to create plots (https://matplotlib.org/)
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

# import time to do animations with plots (https://docs.python.org/3/library/time.html)
import time 


def plot_animation(x,T,Texact,nt,t,dx):
    plt.ion() # enable interactive mode
    fig,ax=plt.subplots(nrows=2,ncols=1, dpi=200)
    T_plot, = ax[0].plot(x, T[:,0], lw=0.8, color='b', linestyle='solid', marker = 'o', mfc='none', ms = 5, label = '$T_{computed}$')
    Texact_plot, = ax[0].plot(x, Texact[:,0], lw=1, color='black', linestyle='dashed', label = '$T_{exact}$')
    ax[0].set_title("t = " + str("%.2f" % t[0]), fontsize = 10, loc='center')
    # ax[0].set_xlim([np.min(x),np.max(x)])
    # ax[0].set_ylim([min(np.min(T[:,0]),np.min(Texact[:,0])),max(np.max(Texact[:,0]),np.max(T[:,0]))])
    ax[0].set_xticks(np.linspace(np.min(x), np.max(x), 10, endpoint=True))
    min_y =  min(np.min(T[:,0]),np.min(Texact[:,0]))
    max_y =  max(np.max(T[:,0]),np.max(Texact[:,0]))
    ax[0].set_yticks(np.linspace(min_y, max_y, 5, endpoint=True))
    #ax[0].set_yticks(np.arange(0, 1, 0.2))
    ax[0].xaxis.set_major_formatter('{x:5.2f}')
    ax[0].yaxis.set_major_formatter('{x:5.2f}')
    ax[0].grid(color='grey', linestyle='solid', linewidth=0.2)
    ax[0].set_xlabel('x', fontsize=10)
    ax[0].set_ylabel('T', fontsize=10)
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1.25), fancybox=True, shadow=False, ncol=5,fontsize = 5) 

    # if shading = "flat", the dimensions of x and x should be one greater than those of C
    cs = ax[1].pcolormesh(np.append(x,x[-1]+dx),[0,0.5,1], [T[:,0],T[:,0]], shading='flat', vmin=min(np.min(T[:,0]),0), vmax=np.max(T[:,0]))
    cbar = fig.colorbar(cs, location='bottom', label='$T_{computed}$')
    ax[1].set_xticks([])
   # ax[1].set_xlim([np.min(x),np.max(x)])
    ax[1].set_yticks([])
    
    fig.tight_layout()
    for jt in range(1,nt):
        T_plot.set_ydata(T[:,jt])
        Texact_plot.set_ydata(Texact[:,jt])
        ax[0].set_title("t = " + str("%.2f" % t[jt]), fontsize = 10, loc='center')
       # ax[0].set_ylim([np.min(T[:,jt]),np.max(Texact[:,jt])])
        min_y =  min(np.min(T[:,jt]),np.min(Texact[:,jt]))
        max_y =  max(np.max(T[:,jt]),np.max(Texact[:,jt]))
        ax[0].set_yticks(np.linspace(min(0,min_y), max(max_y,1), num = 5, endpoint=True))
        if max_y <= 1:
            ax[0].yaxis.set_major_formatter('{x:5.2f}')
        else:
            ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
       
        cs = ax[1].pcolormesh(np.append(x,x[-1]+dx),[0,0.5,1], [T[:,jt],T[:,jt]], shading='flat', vmin=min(np.min(T[:,0]),0), vmax=np.max(T[:,jt]))
        cbar_ticks = np.linspace(min(np.min(T[:,jt]),0), max(np.max(T[:,jt]),1), num=5, endpoint=True)
        cbar.set_ticks(cbar_ticks)
        cbar._draw_all()
        fig.tight_layout()
       
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()
    
        time.sleep(0.05)

    plt.show(block = True)
    
    

def plot_solution_steady(x,T,Texact,Pe,dx):
    plt.ion() # enable interactive mode
    fig,ax=plt.subplots(nrows=2,ncols=1, dpi=200)
    ax[0].plot(x, T, lw=0.8, color='b', linestyle='solid', marker = 'o', mfc='none', ms = 5, label = '$T_{computed}$')
    ax[0].plot(x, Texact, lw=1, color='black', linestyle='dashed', label = '$T_{exact}$')
    ax[0].set_title("Peclet number = " + str("%.2f" % Pe), fontsize = 8, loc='center')
    # ax[0].set_xlim([np.min(x),np.max(x)])
    # ax[0].set_ylim([min(np.min(T),np.min(Texact)),max(np.max(Texact),np.max(T))])
    ax[0].set_xticks(np.linspace(np.min(x), np.max(x), 10, endpoint=True ))
    ax[0].set_yticks(np.linspace(min(np.min(T),np.min(Texact)), max(np.max(Texact),np.max(T)), 5, endpoint=True))
    #ax[0].set_yticks(np.arange(0, 1, 0.2))
    ax[0].xaxis.set_major_formatter('{x:5.2f}')
    ax[0].yaxis.set_major_formatter('{x:5.2f}')
    ax[0].grid(color='grey', linestyle='solid', linewidth=0.2)
    ax[0].set_xlabel('x', fontsize=10, loc='center')
    ax[0].set_ylabel('T', fontsize=10)
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1.25), fancybox=True, shadow=False, ncol=5,fontsize = 5) 

    # if shading = "flat", the dimensions of x and x should be one greater than those of C
    cs = ax[1].pcolormesh(np.append(x,x[-1]+dx),[0,0.5,1], [T,T], shading='flat', vmin=min(np.min(T),0), vmax=np.max(T))
    cbar = fig.colorbar(cs, location='bottom', label='$T_{computed}$')
    ax[1].set_xticks([])
    #ax[1].set_xlim([np.min(x),np.max(x)])
    ax[1].set_yticks([])
    cbar_ticks = np.linspace(min(np.min(T),0), np.max(T), num=5, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    plt.show(block = True)
    

def plot_animation_all(x,T_FTCS,T_CN,T_3l,T_exact,nt,t,dx,Pe,Co,Fo):
    plt.ion() # enable interactive mode
    fig,ax=plt.subplots(nrows=1,ncols=1, dpi=200)
    T_FTCS_plot, = ax.plot(x, T_FTCS[:,0], lw = 0.8, color = 'black', marker = 'p', mfc='none', label = '$T\_FTCS$')
    T_CN_plot, = ax.plot(x, T_CN[:,0], lw = 0.8, color = 'black', marker = 's', mfc='none', label = '$T\_Crank-N.$')
    T_3l_plot, = ax.plot(x, T_3l[:,0], lw = 0.8, color = 'black', marker = 'd', mfc='none', label = '$T\_3level$')
    T_exact_plot, = ax.plot(x, T_exact[:,0], lw=1, color='black', linestyle='dashed', label = '$T\_exact$')
    ax.set_title("Pe_c = " + str("%.2f" % Pe) + ", Fo = " + str("%.2f" % Fo) + ", Co = " + str("%.2f" % Co) + ", t = " + str("%.2f" % t[0]), fontsize = 10, loc='center')
    ax.set_xticks(np.linspace(np.min(x), np.max(x), 10, endpoint=True))
    ax.xaxis.set_major_formatter('{x:5.2f}')
    ax.set_yticks(np.linspace(np.min(T_exact[:,0]), np.max(T_exact[:,0]), 10, endpoint=True))
    #ax[0].set_yticks(np.arange(0, 1, 0.2))
    ax.yaxis.set_major_formatter('{x:5.2f}')
    ax.grid(color='grey', linestyle='solid', linewidth=0.2)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('T', fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.25), fancybox=True, shadow=False, ncol=5,fontsize = 5) 
    for jt in range(1,nt):
        T_FTCS_plot.set_ydata(T_FTCS[:,jt])
        T_CN_plot.set_ydata(T_CN[:,jt])
        T_3l_plot.set_ydata(T_3l[:,jt])
        T_exact_plot.set_ydata(T_exact[:,jt])
        ax.set_title("Pe_c = " + str("%.2f" % Pe) + ", Fo = " + str("%.2f" % Fo) + ", Co = " + str("%.2f" % Co) + ", t = " + str("%.2f" % t[jt]), fontsize = 10, loc='center')
        min_y = min(np.min(T_FTCS[:,jt]),np.min(T_CN[:,jt]),np.min(T_3l[:,jt]),np.min(T_exact[:,jt]))
        max_y = max(np.max(T_FTCS[:,jt]),np.max(T_CN[:,jt]),np.max(T_3l[:,jt]),np.max(T_exact[:,jt]))
        #ax.set_ylim([min_y,max_y])
        ax.set_yticks(np.linspace(min(0,min_y), max_y, num = 10, endpoint=True))
        if max_y <= 1:
            ax.yaxis.set_major_formatter('{x:5.2f}')
        else:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05) 
    plt.show(block = True)