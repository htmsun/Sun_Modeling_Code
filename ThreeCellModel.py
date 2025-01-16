#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:07:46 2024

@author: samantaba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.colors import LinearSegmentedColormap


#%% Defining the two-cell and three-cell models
r = h = 2
v = 1.
c = 1.

def TwoCellModel(t, state, A12, KD):
    N1, N2, D1, D2 = state
    dN1dt = ((A12*D2)**r / (c**r + (A12*D2)**r)) - N1
    dN2dt = ((A12*D1)**r / (c**r + (A12*D1)**r)) - N2
    dD1dt = v*((1 / (1 + (N1/KD)**h)) - D1)
    dD2dt = v*((1 / (1 + (N2/KD)**h)) - D2)
    return [dN1dt, dN2dt, dD1dt, dD2dt]

def ThreeCellModel(t, state, A12, A23, A31, KD):
    N1, N2, N3, D1, D2, D3 = state
    dN1dt = ((A12*D2 + A31*D3)**r / (c**r + (A12*D2 + A31*D3)**r)) - N1
    dN2dt = ((A12*D1 + A23*D3)**r / (c**r + (A12*D1 + A23*D3)**r)) - N2
    dN3dt = ((A31*D1 + A23*D2)**r / (c**r + (A31*D1 + A23*D2)**r)) - N3
    dD1dt = v*((1 / (1 + (N1/KD)**h)) - D1)
    dD2dt = v*((1 / (1 + (N2/KD)**h)) - D2)
    dD3dt = v*((1 / (1 + (N3/KD)**h)) - D3)
    return [dN1dt, dN2dt, dN3dt, dD1dt, dD2dt, dD3dt]
#%% Example of time dynamics of the three cell model
KD = 0.1
A12 = 1.0
A23 = 0.001
A31 = 1.0
p = (A12, A23, A31, KD)
state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
step = 0.01
t = np.arange(0.0, 150.0, step)
result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)

plt.figure()
plt.plot(t, result_odeint [:,0], '-g', linewidth = 2, label = 'N1')
plt.plot(t, result_odeint [:,1], '-c', linewidth = 2, label = 'N2')
plt.plot(t, result_odeint [:,2], '-r', linewidth = 2, label = 'N3')
plt.plot(t, result_odeint [:,3], '--g', linewidth = 4, label = 'D1')
plt.plot(t, result_odeint [:,4], '--c', linewidth = 4, label = 'D2')
plt.plot(t, result_odeint [:,5], '--r', linewidth = 4, label = 'D3')
plt.legend()
plt.xlabel("Simulation time")
plt.ylabel("Notch and Delta levels")
plt.legend()
plt.tight_layout()
plt.show()

#%% Example of phase space of the three cell model
A12 = np.logspace(4.,-2.,21)
KD = np.logspace(-3.,1.,21)
A23 = 0.1
A31 = 0.1
mN1 = np.zeros([len(A12),len(KD)])
mN2 = np.zeros([len(A12),len(KD)])
mN3 = np.zeros([len(A12),len(KD)])
mD1 = np.zeros([len(A12),len(KD)])
mD2 = np.zeros([len(A12),len(KD)])
mD3 = np.zeros([len(A12),len(KD)])
for i in range(len(A12)):
    for j in range(len(KD)):
        p = (A12[i], A23, A31, KD[j])
        state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
        t = np.arange(0.0, 150.0, 0.01)
        result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)
        mN1[i,j] = result_odeint[-1,0]
        mN2[i,j] = result_odeint[-1,1]
        mN3[i,j] = result_odeint[-1,2]
        mD1[i,j] = result_odeint[-1,3]
        mD2[i,j] = result_odeint[-1,4]
        mD3[i,j] = result_odeint[-1,5]
        print(A12[i], KD[j])
#%%
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

X,Y=np.meshgrid(A12,KD)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

im4 = plt.pcolor(1/X,Y,np.matrix.transpose(mD2), cmap='jet', shading='auto')
#im1 = plt.pcolor(X,Y,np.matrix.transpose(mN1), cmap='gray')
#im2 = plt.pcolor(X,Y,np.matrix.transpose(mN2), cmap='gray')
#im3 = plt.pcolor(X,Y,np.matrix.transpose(mD1), cmap='gray')
#im4 = plt.pcolor(X,Y,np.matrix.transpose(mD2), cmap='gray')
#plt.axvline(100, color='red')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('1/A_1-2')
plt.ylabel('KD')
plt.title('A_2-3 = ' +str(A23)+', A_3-1 = '+str(A31))
plt.colorbar(im4, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%% The case of one cell in the middle and two on each side
A12 = np.logspace(4.,-4.,21)
KD = 0.001
A23 = 0
A31 = np.logspace(4.,-4.,21)
mN1 = np.zeros([len(A12),len(A31)])
mN2 = np.zeros([len(A12),len(A31)])
mN3 = np.zeros([len(A12),len(A31)])
mD1 = np.zeros([len(A12),len(A31)])
mD2 = np.zeros([len(A12),len(A31)])
mD3 = np.zeros([len(A12),len(A31)])
for i in range(len(A12)):
    for j in range(len(A31)):
        p = (A12[i], A23, A31[j], KD)
        state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
        t = np.arange(0.0, 150.0, 0.01)
        result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)
        mN1[i,j] = result_odeint[-1,0]
        mN2[i,j] = result_odeint[-1,1]
        mN3[i,j] = result_odeint[-1,2]
        mD1[i,j] = result_odeint[-1,3]
        mD2[i,j] = result_odeint[-1,4]
        mD3[i,j] = result_odeint[-1,5]
        print(A12[i], A31[j])


#%% Final Notch Delta Levels Heatmap
X,Y=np.meshgrid(1/A12,1/A31)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', '#A8CB63'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(221)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mN1), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax1.pcolor(X,Y,np.matrix.transpose(mN1), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Final Notch Level cell 1')

ax2 = plt.subplot(222)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mN2), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax2.pcolor(X,Y,np.matrix.transpose(mN2), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Final Notch Level cell 2')

ax3 = plt.subplot(223)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mD1), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mD1), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Final Delta Level cell 1')

ax4 = plt.subplot(224)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mD2), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax4.pcolor(X,Y,np.matrix.transpose(mD2), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Final Delta Level cell 2')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')

ax1.set_xlabel('K Notch Activation')
ax1.set_ylabel('K Delta Inhibition')
ax2.set_xlabel('K Notch Activation')
ax2.set_ylabel('K Delta Inhibition')
ax3.set_xlabel('K Notch Activation')
ax3.set_ylabel('K Delta Inhibition')
ax4.set_xlabel('K Notch Activation')
ax4.set_ylabel('K Delta Inhibition')

#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()

#%% The case of one cell in the middle and two on each side
A12 = np.logspace(4.,-2.,101)
KD = np.logspace(-3.,1.,101)
#A12 = np.logspace(4.,-4.,101)
A23 = 0
A31 = np.array([0, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000])
mN1_r = np.zeros([len(A31),len(A12),len(KD)])
mN2_r = np.zeros([len(A31),len(A12),len(KD)])
mN3_r = np.zeros([len(A31),len(A12),len(KD)])
mD1_r = np.zeros([len(A31),len(A12),len(KD)])
mD2_r = np.zeros([len(A31),len(A12),len(KD)])
mD3_r = np.zeros([len(A31),len(A12),len(KD)])

for k in range(len(A31)):
    for i in range(len(A12)):
        for j in range(len(KD)):
            p = (A12[i], A23, A31[k], KD[j])
            state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
            t = np.arange(0.0, 500.0, 0.01)
            result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)
            mN1_r[k,i,j] = result_odeint[-1,0]
            mN2_r[k,i,j] = result_odeint[-1,1]
            mN3_r[k,i,j] = result_odeint[-1,2]
            mD1_r[k,i,j] = result_odeint[-1,3]
            mD2_r[k,i,j] = result_odeint[-1,4]
            mD3_r[k,i,j] = result_odeint[-1,5]
        print(A31[k])
#%% Final Delta level
X,Y=np.meshgrid(1/A12,KD)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', '#A8CB63'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax1.pcolor(X,Y,np.matrix.transpose(mD1_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Final Delta Level cell 1')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax2.pcolor(X,Y,np.matrix.transpose(mD1_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Final Delta Level cell 1')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mD1_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Final Delta Level cell 1')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[3,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax4.pcolor(X,Y,np.matrix.transpose(mD1_r[3,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Final Delta Level cell 1')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[4,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax5.pcolor(X,Y,np.matrix.transpose(mD1_r[4,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Final Delta Level cell 1')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[5,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax6.pcolor(X,Y,np.matrix.transpose(mD1_r[5,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Final Delta Level cell 1')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[6,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax7.pcolor(X,Y,np.matrix.transpose(mD1_r[6,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Final Delta Level cell 1')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[7,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax8.pcolor(X,Y,np.matrix.transpose(mD1_r[7,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Final Delta Level cell 1')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[8,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax9.pcolor(X,Y,np.matrix.transpose(mD1_r[8,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Final Delta Level cell 1')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('K_D')
ax2.set_ylabel('K_D')
ax3.set_ylabel('K_D')
ax4.set_ylabel('K_D')
ax5.set_ylabel('K_D')
ax6.set_ylabel('K_D')
ax7.set_ylabel('K_D')
ax8.set_ylabel('K_D')
ax9.set_ylabel('K_D')

ax1.title.set_text('A_1-3 = '+str(A31[0]))
ax2.title.set_text('A_1-3 = '+str(A31[1]))
ax3.title.set_text('A_1-3 = '+str(A31[2]))
ax4.title.set_text('A_1-3 = '+str(A31[3]))
ax5.title.set_text('A_1-3 = '+str(A31[4]))
ax6.title.set_text('A_1-3 = '+str(A31[5]))
ax7.title.set_text('A_1-3 = '+str(A31[6]))
ax8.title.set_text('A_1-3 = '+str(A31[7]))
ax9.title.set_text('A_1-3 = '+str(A31[8]))


#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%% Final Notch level
X,Y=np.meshgrid(1/A12,KD)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', '#A8CB63'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax1.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Final Notch Level cell 3')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax2.pcolor(X,Y,np.matrix.transpose(mN3_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Final Notch Level cell 3')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mN3_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Final Notch Level cell 3')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[3,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax4.pcolor(X,Y,np.matrix.transpose(mN3_r[3,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Final Notch Level cell 3')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[4,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax5.pcolor(X,Y,np.matrix.transpose(mN3_r[4,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Final Notch Level cell 3')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[5,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax6.pcolor(X,Y,np.matrix.transpose(mN3_r[5,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Final Notch Level cell 3')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[6,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax7.pcolor(X,Y,np.matrix.transpose(mN3_r[6,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Final Notch Level cell 3')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[7,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax8.pcolor(X,Y,np.matrix.transpose(mN3_r[7,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Final Notch Level cell 3')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[8,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax9.pcolor(X,Y,np.matrix.transpose(mN3_r[8,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Final Notch Level cell 3')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('K_D')
ax2.set_ylabel('K_D')
ax3.set_ylabel('K_D')
ax4.set_ylabel('K_D')
ax5.set_ylabel('K_D')
ax6.set_ylabel('K_D')
ax7.set_ylabel('K_D')
ax8.set_ylabel('K_D')
ax9.set_ylabel('K_D')

ax1.title.set_text('A_1-3 = '+str(A31[0]))
ax2.title.set_text('A_1-3 = '+str(A31[1]))
ax3.title.set_text('A_1-3 = '+str(A31[2]))
ax4.title.set_text('A_1-3 = '+str(A31[3]))
ax5.title.set_text('A_1-3 = '+str(A31[4]))
ax6.title.set_text('A_1-3 = '+str(A31[5]))
ax7.title.set_text('A_1-3 = '+str(A31[6]))
ax8.title.set_text('A_1-3 = '+str(A31[7]))
ax9.title.set_text('A_1-3 = '+str(A31[8]))


#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%% Calculating Notch speed 
t = np.arange(0.0, 50.0, 0.01)
A12 = np.logspace(4.,-2.,101)
KD = np.logspace(-3.,1.,101)
A23 = 0
A31 = np.array([0, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000])
mN1_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
mN2_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
mN3_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
GFPN1 = np.zeros([len(t)])
GFPN2 = np.zeros([len(t)])
GFPN3 = np.zeros([len(t)])

for k in range(len(A31)):
    for i in range(len(A12)):
        for j in range(len(KD)):
            p = (A12[i], A23, A31[k], KD[j])
            state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
            t = np.arange(0.0, 50.0, 0.01)
            result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)
            for l in range(1,len(t)):
                GFPN1[l] = GFPN1[l-1] + result_odeint[l,0]
                GFPN2[l] = GFPN2[l-1] + result_odeint[l,1]
                GFPN3[l] = GFPN3[l-1] + result_odeint[l,2]
            mN1_t[k,i,j,:] = GFPN1
            mN2_t[k,i,j,:] = GFPN2
            mN3_t[k,i,j,:] = GFPN3
        print(A31[k])

mNG1 = np.zeros([len(A31),len(A12),len(KD)])
mNG2 = np.zeros([len(A31),len(A12),len(KD)])
mNG3 = np.zeros([len(A31),len(A12),len(KD)])
ti = 0
tf = ti + 1200
for k in range(len(A31)):
    for i in range(len(A12)):
        for j in range(len(KD)):
            mNG1[k,i,j] = (mN1_t[k,i,j,tf] - mN1_t[k,i,j,ti]) / (t[tf] - t[ti])
            mNG2[k,i,j] = (mN2_t[k,i,j,tf] - mN2_t[k,i,j,ti]) / (t[tf] - t[ti])
            mNG3[k,i,j] = (mN3_t[k,i,j,tf] - mN3_t[k,i,j,ti]) / (t[tf] - t[ti])


#%% Notch activation speed 

#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', 'green'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
vMax = 100

X,Y=np.meshgrid(1/A12,KD)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax1.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Average Speed Cell 3')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax2.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Average Speed Cell 3')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax3.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Average Speed Cell 3')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[3,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax4.pcolor(X,Y,np.matrix.transpose(mNG3[3,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Average Speed Cell 3')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[4,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax5.pcolor(X,Y,np.matrix.transpose(mNG3[4,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Average Speed Cell 3')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[5,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax6.pcolor(X,Y,np.matrix.transpose(mNG3[5,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Average Speed Cell 3')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[6,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax7.pcolor(X,Y,np.matrix.transpose(mNG3[6,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Average Speed Cell 3')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[7,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax8.pcolor(X,Y,np.matrix.transpose(mNG3[7,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Average Speed Cell 3')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[8,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax9.pcolor(X,Y,np.matrix.transpose(mNG3[8,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Average Speed Cell 3')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('K_D')
ax2.set_ylabel('K_D')
ax3.set_ylabel('K_D')
ax4.set_ylabel('K_D')
ax5.set_ylabel('K_D')
ax6.set_ylabel('K_D')
ax7.set_ylabel('K_D')
ax8.set_ylabel('K_D')
ax9.set_ylabel('K_D')

ax1.title.set_text('A_1-3 = '+str(A31[0]))
ax2.title.set_text('A_1-3 = '+str(A31[1]))
ax3.title.set_text('A_1-3 = '+str(A31[2]))
ax4.title.set_text('A_1-3 = '+str(A31[3]))
ax5.title.set_text('A_1-3 = '+str(A31[4]))
ax6.title.set_text('A_1-3 = '+str(A31[5]))
ax7.title.set_text('A_1-3 = '+str(A31[6]))
ax8.title.set_text('A_1-3 = '+str(A31[7]))
ax9.title.set_text('A_1-3 = '+str(A31[8]))


#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%% 
t = np.arange(0.0, 50.0, 0.01)
A12 = np.logspace(4.,-2.,101)
KD = np.array([0.1,0.5,1.0])
A23 = 0
A31 = np.logspace(4.,-2.,101)
mN1_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
mN2_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
mN3_t = np.zeros([len(A31),len(A12),len(KD),len(t)])
GFPN1 = np.zeros([len(t)])
GFPN2 = np.zeros([len(t)])
GFPN3 = np.zeros([len(t)])

for k in range(len(KD)):
    for i in range(len(A12)):
        for j in range(len(A31)):
            p = (A12[i], A23, A31[j], KD[k])
            state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
            t = np.arange(0.0, 50.0, 0.01)
            result_odeint = odeint(ThreeCellModel, state0, t, p, tfirst=True)
            for l in range(1,len(t)):
                GFPN1[l] = GFPN1[l-1] + result_odeint[l,0]
                GFPN2[l] = GFPN2[l-1] + result_odeint[l,1]
                GFPN3[l] = GFPN3[l-1] + result_odeint[l,2]
            mN1_t[j,i,k,:] = GFPN1
            mN2_t[j,i,k,:] = GFPN2
            mN3_t[j,i,k,:] = GFPN3
        print(KD[k])
#%%
mNG1 = np.zeros([len(KD),len(A31),len(A12)])
mNG2 = np.zeros([len(KD),len(A31),len(A12)])
mNG3 = np.zeros([len(KD),len(A31),len(A12)])
ti = 0
tf = ti + 1200
for k in range(len(KD)):
    for i in range(len(A12)):
        for j in range(len(A31)):
            mNG1[k,i,j] = (mN1_t[j,i,k,tf] - mN1_t[j,i,k,ti]) / (t[tf] - t[ti])
            mNG2[k,i,j] = (mN2_t[j,i,k,tf] - mN2_t[j,i,k,ti]) / (t[tf] - t[ti])
            mNG3[k,i,j] = (mN3_t[j,i,k,tf] - mN3_t[j,i,k,ti]) / (t[tf] - t[ti])


#%% Final Notch Delta Levels Heatmap for all a and b

#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', 'green'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
vMax = 100

X,Y=np.meshgrid(1/A12,1/A31)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax1.pcolor(X,Y,np.matrix.transpose(mNG1[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Average Speed Cell 1')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax2.pcolor(X,Y,np.matrix.transpose(mNG1[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Average Speed Cell 1')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax3.pcolor(X,Y,np.matrix.transpose(mNG1[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Average Speed Cell 1')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax4.pcolor(X,Y,np.matrix.transpose(mNG2[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Average Speed Cell 2')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax5.pcolor(X,Y,np.matrix.transpose(mNG2[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Average Speed Cell 2')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax6.pcolor(X,Y,np.matrix.transpose(mNG2[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Average Speed Cell 2')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax7.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Average Speed Cell 3')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax8.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Average Speed Cell 3')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax9.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Average Speed Cell 3')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('1/A 1-3')
ax2.set_ylabel('1/A 1-3')
ax3.set_ylabel('1/A 1-3')
ax4.set_ylabel('1/A 1-3')
ax5.set_ylabel('1/A 1-3')
ax6.set_ylabel('1/A 1-3')
ax7.set_ylabel('1/A 1-3')
ax8.set_ylabel('1/A 1-3')
ax9.set_ylabel('1/A 1-3')

ax1.title.set_text('K_D = '+str(KD[0]))
ax2.title.set_text('K_D = '+str(KD[1]))
ax3.title.set_text('K_D = '+str(KD[2]))
ax4.title.set_text('K_D = '+str(KD[0]))
ax5.title.set_text('K_D = '+str(KD[1]))
ax6.title.set_text('K_D = '+str(KD[2]))
ax7.title.set_text('K_D = '+str(KD[0]))
ax8.title.set_text('K_D = '+str(KD[1]))
ax9.title.set_text('K_D = '+str(KD[2]))

#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()



#%% Three cell model where all cells are interacting
A12 = np.logspace(4.,-2.,51)
KD = np.array([0.1,0.5,1.0])
A23 = np.array([0.1,1,10])
A31 = np.logspace(4.,-2.,51)
mN1_r = np.zeros([len(KD),len(A31),len(A12)])
mN2_r = np.zeros([len(KD),len(A31),len(A12)])
mN3_r = np.zeros([len(KD),len(A31),len(A12)])
mD1_r = np.zeros([len(KD),len(A31),len(A12)])
mD2_r = np.zeros([len(KD),len(A31),len(A12)])
mD3_r = np.zeros([len(KD),len(A31),len(A12)])

for k in range(len(KD)):
    for i in range(len(A12)):
        for j in range(len(A31)):
            p1 = (A12[i], A23[0], A31[j], KD[k])
            p2 = (A12[i], A23[1], A31[j], KD[k])
            p3 = (A12[i], A23[2], A31[j], KD[k])
            state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
            t = np.arange(0.0, 150.0, 0.01)
            result_odeint1 = odeint(ThreeCellModel, state0, t, p1, tfirst=True)
            result_odeint2 = odeint(ThreeCellModel, state0, t, p2, tfirst=True)
            result_odeint3 = odeint(ThreeCellModel, state0, t, p3, tfirst=True)
            mN1_r[k,j,i] = result_odeint1[-1,0]
            mN2_r[k,j,i] = result_odeint2[-1,0]
            mN3_r[k,j,i] = result_odeint3[-1,0]
            mD1_r[k,j,i] = result_odeint1[-1,3]
            mD2_r[k,j,i] = result_odeint2[-1,3]
            mD3_r[k,j,i] = result_odeint3[-1,3]
    print(KD[k])
#%%
X,Y=np.meshgrid(1/A12,1/A31)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', '#A8CB63'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mN1_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax1.pcolor(X,Y,np.matrix.transpose(mN1_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Final Notch Level cell 1')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mN1_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax2.pcolor(X,Y,np.matrix.transpose(mN1_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Final Notch Level cell 1')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mN1_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mN1_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Final Notch Level cell 1')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax4.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Final Notch Level cell 1')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mN2_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax5.pcolor(X,Y,np.matrix.transpose(mN2_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Final Notch Level cell 1')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mN2_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax6.pcolor(X,Y,np.matrix.transpose(mN2_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Final Notch Level cell 1')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax7.pcolor(X,Y,np.matrix.transpose(mN3_r[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Final Notch Level cell 1')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax8.pcolor(X,Y,np.matrix.transpose(mN3_r[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Final Notch Level cell 1')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mN3_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
ax9.pcolor(X,Y,np.matrix.transpose(mN3_r[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Final Notch Level cell 1')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('1/A 1-3')
ax2.set_ylabel('1/A 1-3')
ax3.set_ylabel('1/A 1-3')
ax4.set_ylabel('1/A 1-3')
ax5.set_ylabel('1/A 1-3')
ax6.set_ylabel('1/A 1-3')
ax7.set_ylabel('1/A 1-3')
ax8.set_ylabel('1/A 1-3')
ax9.set_ylabel('1/A 1-3')

ax1.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[0]))
ax2.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[0]))
ax3.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[0]))
ax4.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[1]))
ax5.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[1]))
ax6.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[1]))
ax7.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[2]))
ax8.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[2]))
ax9.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[2]))


#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%%
X,Y=np.meshgrid(1/A12,1/A31)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', '#A8CB63'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax1.pcolor(X,Y,np.matrix.transpose(mD1_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Final Delta Level cell 1')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax2.pcolor(X,Y,np.matrix.transpose(mD1_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Final Delta Level cell 1')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mD1_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax3.pcolor(X,Y,np.matrix.transpose(mD1_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Final Delta Level cell 1')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mD2_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax4.pcolor(X,Y,np.matrix.transpose(mD2_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Final Delta Level cell 1')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mD2_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax5.pcolor(X,Y,np.matrix.transpose(mD2_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Final Delta Level cell 1')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mD2_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax6.pcolor(X,Y,np.matrix.transpose(mD2_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Final Delta Level cell 1')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mD3_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax7.pcolor(X,Y,np.matrix.transpose(mD3_r[0,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Final Delta Level cell 1')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mD3_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax8.pcolor(X,Y,np.matrix.transpose(mD3_r[1,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Final Delta Level cell 1')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mD3_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
ax9.pcolor(X,Y,np.matrix.transpose(mD3_r[2,:,:]), cmap=Deltacm, shading='auto', vmin=0, vmax=1)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Final Delta Level cell 1')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('1/A 1-3')
ax2.set_ylabel('1/A 1-3')
ax3.set_ylabel('1/A 1-3')
ax4.set_ylabel('1/A 1-3')
ax5.set_ylabel('1/A 1-3')
ax6.set_ylabel('1/A 1-3')
ax7.set_ylabel('1/A 1-3')
ax8.set_ylabel('1/A 1-3')
ax9.set_ylabel('1/A 1-3')

ax1.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[0]))
ax2.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[0]))
ax3.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[0]))
ax4.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[1]))
ax5.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[1]))
ax6.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[1]))
ax7.title.set_text('K_D = '+str(KD[0])+'A_2-3 = '+str(A23[2]))
ax8.title.set_text('K_D = '+str(KD[1])+'A_2-3 = '+str(A23[2]))
ax9.title.set_text('K_D = '+str(KD[2])+'A_2-3 = '+str(A23[2]))


#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
#%% Calculating Notch activation speed for the three cell model where all cells are interacting 
t = np.arange(0.0, 50.0, 0.01)

A12 = np.logspace(4.,-2.,51)
KD = np.array([0.1,0.5,1.0])
A23 = np.array([0.1,1,10])
A31 = np.logspace(4.,-2.,51)
mN1_r = np.zeros([len(KD),len(A31),len(A12),len(t)])
mN2_r = np.zeros([len(KD),len(A31),len(A12),len(t)])
mN3_r = np.zeros([len(KD),len(A31),len(A12),len(t)])
GFPN1 = np.zeros([len(t)])
GFPN2 = np.zeros([len(t)])
GFPN3 = np.zeros([len(t)])

for k in range(len(KD)):
    for i in range(len(A12)):
        for j in range(len(A31)):
            p1 = (A12[i], A23[0], A31[j], KD[k])
            p2 = (A12[i], A23[1], A31[j], KD[k])
            p3 = (A12[i], A23[2], A31[j], KD[k])
            state0 = [0.11, 0.1, 0.1, 0.9, 0.9, 0.9]
            result_odeint1 = odeint(ThreeCellModel, state0, t, p1, tfirst=True)
            result_odeint2 = odeint(ThreeCellModel, state0, t, p2, tfirst=True)
            result_odeint3 = odeint(ThreeCellModel, state0, t, p3, tfirst=True)
            for l in range(1,len(t)):
                GFPN1[l] = GFPN1[l-1] + result_odeint1[l,0]
                GFPN2[l] = GFPN2[l-1] + result_odeint2[l,0]
                GFPN3[l] = GFPN3[l-1] + result_odeint3[l,0]
            mN1_r[k,j,i,:] = GFPN1
            mN2_r[k,j,i,:] = GFPN2
            mN3_r[k,j,i,:] = GFPN3
    print(KD[k])
#%%
mNG1 = np.zeros([len(KD),len(A31),len(A12)])
mNG2 = np.zeros([len(KD),len(A31),len(A12)])
mNG3 = np.zeros([len(KD),len(A31),len(A12)])
ti = 0
tf = ti + 1200
for k in range(len(KD)):
    for i in range(len(A12)):
        for j in range(len(A31)):
            mNG1[k,j,i] = (mN1_r[k,j,i,tf] - mN1_r[k,j,i,ti]) / (t[tf] - t[ti])
            mNG2[k,j,i] = (mN2_r[k,j,i,tf] - mN2_r[k,j,i,ti]) / (t[tf] - t[ti])
            mNG3[k,j,i] = (mN3_r[k,j,i,tf] - mN3_r[k,j,i,ti]) / (t[tf] - t[ti])
#%%

#NotchColor = ['white', '#8EB644'] 
NotchColor = ['white', 'green'] 
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
vMax = 100

X,Y=np.meshgrid(1/A12,1/A31)
z = np.logspace(np.log10(10), np.log10(1000), 5)
Z=np.vstack((z,z))

#DeltaColor = ['white', '#93bdc5']
DeltaColor = ['white', '#93bdc5']
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
#NotchColor = ['white', '#8EB644'] 
fig = plt.figure(figsize=(13,10))
ax1 = plt.subplot(331)
im1 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax1.pcolor(X,Y,np.matrix.transpose(mNG1[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im1, ax=ax1, orientation='vertical', label = 'Average Speed Cell 1')

ax2 = plt.subplot(332)
im2 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax2.pcolor(X,Y,np.matrix.transpose(mNG1[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im2, ax=ax2, orientation='vertical', label = 'Average Speed Cell 1')

ax3 = plt.subplot(333)
im3 = plt.pcolor(X,Y,np.matrix.transpose(mNG1[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax3.pcolor(X,Y,np.matrix.transpose(mNG1[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im3, ax=ax3, orientation='vertical', label = 'Average Speed Cell 1')

ax4 = plt.subplot(334)
im4 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax4.pcolor(X,Y,np.matrix.transpose(mNG2[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im4, ax=ax4, orientation='vertical', label = 'Average Speed Cell 1')

ax5 = plt.subplot(335)
im5 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax5.pcolor(X,Y,np.matrix.transpose(mNG2[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im5, ax=ax5, orientation='vertical', label = 'Average Speed Cell 1')

ax6 = plt.subplot(336)
im6 = plt.pcolor(X,Y,np.matrix.transpose(mNG2[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax6.pcolor(X,Y,np.matrix.transpose(mNG2[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im6, ax=ax6, orientation='vertical', label = 'Average Speed Cell 1')

ax7 = plt.subplot(337)
im7 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax7.pcolor(X,Y,np.matrix.transpose(mNG3[0,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im7, ax=ax7, orientation='vertical', label = 'Average Speed Cell 1')

ax8 = plt.subplot(338)
im8 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax8.pcolor(X,Y,np.matrix.transpose(mNG3[1,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im8, ax=ax8, orientation='vertical', label = 'Average Speed Cell 1')

ax9 = plt.subplot(339)
im9 = plt.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
ax9.pcolor(X,Y,np.matrix.transpose(mNG3[2,:,:]), cmap=Notchcm, shading='auto', vmin=0, vmax=vMax)
plt.colorbar(im9, ax=ax9, orientation='vertical', label = 'Average Speed Cell 1')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_xscale('log')
ax8.set_xscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_xscale('log')
ax9.set_yscale('log')

ax1.set_xlabel('1/A 1-2')
ax2.set_xlabel('1/A 1-2')
ax3.set_xlabel('1/A 1-2')
ax4.set_xlabel('1/A 1-2')
ax5.set_xlabel('1/A 1-2')
ax6.set_xlabel('1/A 1-2')
ax7.set_xlabel('1/A 1-2')
ax8.set_xlabel('1/A 1-2')
ax9.set_xlabel('1/A 1-2')
ax1.set_ylabel('1/A 1-3')
ax2.set_ylabel('1/A 1-3')
ax3.set_ylabel('1/A 1-3')
ax4.set_ylabel('1/A 1-3')
ax5.set_ylabel('1/A 1-3')
ax6.set_ylabel('1/A 1-3')
ax7.set_ylabel('1/A 1-3')
ax8.set_ylabel('1/A 1-3')
ax9.set_ylabel('1/A 1-3')

ax1.title.set_text('K_D = '+str(KD[0])+', A_2-3 = '+str(A23[0]))
ax2.title.set_text('K_D = '+str(KD[1])+', A_2-3 = '+str(A23[0]))
ax3.title.set_text('K_D = '+str(KD[2])+', A_2-3 = '+str(A23[0]))
ax4.title.set_text('K_D = '+str(KD[0])+', A_2-3 = '+str(A23[1]))
ax5.title.set_text('K_D = '+str(KD[1])+', A_2-3 = '+str(A23[1]))
ax6.title.set_text('K_D = '+str(KD[2])+', A_2-3 = '+str(A23[1]))
ax7.title.set_text('K_D = '+str(KD[0])+', A_2-3 = '+str(A23[2]))
ax8.title.set_text('K_D = '+str(KD[1])+', A_2-3 = '+str(A23[2]))
ax9.title.set_text('K_D = '+str(KD[2])+', A_2-3 = '+str(A23[2]))

#plt.colorbar(im1, orientation='horizontal', label = 'Final Delta Level cell 2')
plt.tight_layout()
plt.show()
