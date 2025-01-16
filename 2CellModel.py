#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:53:32 2024
Updated Jan 14 2025

@author: samantaba
"""
import numpy as np
import pandas as pd
import statistics as st
import seaborn as sns
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

#%% Defining Guisoni Model
r = h = 2
v = 1.

def GuisoniModel(t, state, a, b):
    N1, N2, D1, D2 = state
    dN1dt = (D2**r / (a**r + D2**r)) - N1
    dN2dt = (D1**r / (a**r + D1**r)) - N2
    dD1dt = v*((1 / (1 + (N1/b)**h)) - D1)
    dD2dt = v*((1 / (1 + (N2/b)**h)) - D2)
    return [dN1dt, dN2dt, dD1dt, dD2dt]


#%% Solving the model for accumulated GFP (Notch speed readout)
a = np.logspace(-2.,1.,101)
b = np.logspace(-2.,1.,101)
t = np.arange(0.0, 50.0, 0.01)
mN1 = np.zeros([len(a),len(b), len(t)])
mN2 = np.zeros([len(a),len(b), len(t)])
GFPN1 = np.zeros([len(t)])
GFPN2 = np.zeros([len(t)])
for i in range(len(a)):
    for j in range(len(b)):
        p = (a[i], b[j])
        state0 = [0.011, 0.01, 0.99, 0.99]
        result_odeint = odeint(GuisoniModel, state0, t, p, tfirst=True)
        for k in range(1,len(t)):
            GFPN1[k] = GFPN1[k-1] + result_odeint[k,0]
            GFPN2[k] = GFPN2[k-1] + result_odeint[k,1]
        mN1[i,j,:] = GFPN1
        mN2[i,j,:] = GFPN2
        print([i,j])

mNG1 = np.zeros([len(a),len(b)])
mNG2 = np.zeros([len(a),len(b)])
ti = 400
tf = 1000
for i in range(len(a)):
    for j in range(len(b)):
        mNG1[i,j] = (mN1[i,j,tf] - mN1[i,j,ti]) / (t[tf] - t[ti])
        mNG2[i,j] = (mN2[i,j,tf] - mN2[i,j,ti]) / (t[tf] - t[ti])


#%%  Plotting Notch speed heatmap
X,Y=np.meshgrid(a,b)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))
NotchColor = ['white', '#8EB644'] 
#NotchColor = ['white', '#006400'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(7,5))
ax2 = plt.subplot(111)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mNG1), cmap=Notchcm, shading='auto')
ax2.pcolor(X,Y,np.matrix.transpose(mNG1), cmap=Notchcm, shading='auto')
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Average Speed')
#plt.title('t='+str(ti)+'-'+str(tf))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation')
ax2.set_ylabel('K Delta Inhibition')
plt.tight_layout()
plt.show()


#%% Final Delta Levels Heatmap for all KD and KN
a = np.logspace(-2.,1.,101)
b = np.logspace(-2.,1.,101)
t = np.arange(0.0, 50.0, 0.01)
mN1 = np.zeros([len(a),len(b), len(t)])
mN2 = np.zeros([len(a),len(b), len(t)])
mD1 = np.zeros([len(a),len(b), len(t)])
mD2 = np.zeros([len(a),len(b), len(t)])
for i in range(len(a)):
    for j in range(len(b)):
        p = (a[i], b[j])
        state0 = [0.011, 0.01, 0.99, 0.99]
        result_odeint = odeint(GuisoniModel, state0, t, p, tfirst=True)
        mN1[i,j,:] = result_odeint[:,0]
        mN2[i,j,:] = result_odeint[:,1]
        mD1[i,j,:] = result_odeint[:,2]
        mD2[i,j,:] = result_odeint[:,3]
        print([i,j])

tn= 1000
mD = mD1[:,:,tn]

#%% Plotting Delta levels heatmap
X,Y=np.meshgrid(a,b)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))
#DeltaColor = ['white', '#366C77']
DeltaColor = ['white', '#93bdc5']
#DeltaColor = ['white', '#93bdc5', '#366C77']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
ax3 = plt.subplot(111)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mD), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mD), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Delta Level')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('K Notch Activation')
ax3.set_ylabel('K Delta Inhibition')
plt.tight_layout()
plt.show()

#%% Plotting Notch levels heatmap
tn= 1200
mN = mN1[:,:,tn]

X,Y=np.meshgrid(a,b)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))
#DeltaColor = ['white', '#366C77']
#DeltaColor = ['white', '#93bdc5']
NotchColor = ['white', 'green']
#DeltaColor = ['white', '#93bdc5', '#366C77']
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
ax3 = plt.subplot(111)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mN), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mN), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Notch Level')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('K Notch Activation')
ax3.set_ylabel('K Delta Inhibition')
plt.tight_layout()
plt.show()


#%% Showing specific time dynamics
a = 0.5
b = np.array([0.01,0.1, 0.2, 0.3, 0.4])
state0 = [0.011, 0.01, 0.99, 0.99]
t = np.arange(0.0, 150.0, 0.01)
mN1 = np.zeros([len(t),len(b)])
mD1 = np.zeros([len(t),len(b)])
mN2 = np.zeros([len(t),len(b)])
mD2 = np.zeros([len(t),len(b)])
GFP = np.zeros([len(t),len(b)])

for i in range(len(b)):
    p = (a, b[i])
    result_odeint = odeint(GuisoniModel, state0, t, p, tfirst=True)
    mN1[:,i] = result_odeint[:,0]
    mD1[:,i] = result_odeint[:,2]
    mN2[:,i] = result_odeint[:,1]
    mD2[:,i] = result_odeint[:,3]
for j in range(len(b)):
    for i in range(1,len(t)):
        GFP[i, j] = GFP[i-1, j] + mN1[i,j]
        
#%%Plotting time evolution plots
ti = 0
tf = 1200
plt.figure()
plt.plot(t[ti:tf],GFP[ti:tf,0], lw = 3, color = 'green', alpha = 0.2)
plt.plot(t[ti:tf],GFP[ti:tf,1], lw = 3, color = 'green', alpha = 0.4)
plt.plot(t[ti:tf],GFP[ti:tf,2], lw = 3, color = 'green', alpha = 0.6)
plt.plot(t[ti:tf],GFP[ti:tf,3], lw = 3, color = 'green', alpha = 0.8)
plt.plot(t[ti:tf],GFP[ti:tf,4], lw = 3, color = 'green', alpha = 1.0)
#plt.ylabel('Expected Accumulated GFP level')
plt.xlabel('Time')
plt.title('KN = 0.5,' + ' KD ='+str(b))
plt.xlim(0,12)
plt.show()
#%% Plotting Delta and Notch speed dynamics together
GFPcolor1 = '#8EB644'
GFPcolor2 = '#006400'
Dlcolor1 = '#93bdc5'
Dlcolor2 = '#366C77'
plt.figure(figsize=(11,5))
ax1 = plt.subplot(121)
ax2 = ax1.twinx()
ax1.plot(t[ti:tf],mD1[ti:tf,0], lw = 3, color = Dlcolor1, alpha = 1.0)
ax2.plot(t[ti:tf],GFP[ti:tf,0], lw = 3, color = GFPcolor1, alpha = 1.0)
#ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax1.set_title("low KD")
ax3 = plt.subplot(122)
ax4 = ax3.twinx()  
ax3.plot(t[ti:tf],mD1[ti:tf,4], lw = 3, color = Dlcolor1, alpha = 1.0)
ax3.set_title("High KD")
ax4.plot(t[ti:tf],GFP[ti:tf,4], lw = 3, color = GFPcolor1, alpha = 1.0)
ax1.set_ylim(0,1)
ax3.set_ylim(0,1)
ax2.set_ylim(0,600)
ax4.set_ylim(0,600)
ax1.set_xlim(0,12)
ax3.set_xlim(0,12)
ax1.set_ylabel('Delta Levels')
ax3.set_ylabel('Delta Levels')
ax2.set_ylabel('Nocth Speed')
ax4.set_ylabel('Notch Speed')
ax1.set_xlabel('Relative Time')
ax3.set_xlabel('Relative Time')
ax1.yaxis.label.set_color(Dlcolor2)  
ax3.yaxis.label.set_color(Dlcolor2)  
ax2.yaxis.label.set_color(GFPcolor2)  
ax4.yaxis.label.set_color(GFPcolor2)  
plt.tight_layout()
plt.show()