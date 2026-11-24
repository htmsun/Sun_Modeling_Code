#!/usr/bin/env python
# coding: utf-8

# In[23]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:07:46 2024

@author: samantaba

edited by Miriam Nov 19 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'     # keeps text editable in Illustrator
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'

from scipy.integrate import odeint
from scipy.signal import lfilter
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.colors import LinearSegmentedColormap


r = h = 2
v = 1.
c = 1.
# note A = c/K_N

ALPHA = 0.1        # reporter degradation rate (dimensionless = tau_N / tau_reporter)
REL_TOL = 1e-5

def reporter(N, dt, alpha, beta=1.0):
    """Solve dR/dt = beta*N - alpha*R with R(0) = 0."""
    if alpha == 0:
        R = np.cumsum(N) * dt * beta
        return R - R[0]
    decay = np.exp(-alpha * dt)
    return lfilter([0.0, 1.0], [1.0, -decay], beta * N * (1.0 - decay) / alpha)

def steady_state_idx(arr, rel_tol=REL_TOL):
    final_val = arr[-1]
    threshold = rel_tol * max(abs(final_val), 1e-8)
    beyond = np.where(np.abs(arr - final_val) >= threshold)[0]
    if beyond.size == 0:
        return 0
    return min(int(beyond[-1]) + 1, len(arr) - 1)


# In[24]:


def TwoCellModel(t, state, A12, KD):
    N1, N2, D1, D2 = state
    dN1dt = ((A12*D2)**r / (c**r + (A12*D2)**r)) - N1
    dN2dt = ((A12*D1)**r / (c**r + (A12*D1)**r)) - N2
    dD1dt = v*((1 / (1 + (N1/KD)**h)) - D1)
    dD2dt = v*((1 / (1 + (N2/KD)**h)) - D2)
    return [dN1dt, dN2dt, dD1dt, dD2dt]


# In[25]:


def ThreeCellModel(t, state, A12, A23, A31, KD):
    N1, N2, N3, D1, D2, D3 = state
    dN1dt = ((A12*D2 + A31*D3)**r / (c**r + (A12*D2 + A31*D3)**r)) - N1
    dN2dt = ((A12*D1 + A23*D3)**r / (c**r + (A12*D1 + A23*D3)**r)) - N2
    dN3dt = ((A31*D1 + A23*D2)**r / (c**r + (A31*D1 + A23*D2)**r)) - N3
    dD1dt = v*((1 / (1 + (N1/KD)**h)) - D1)
    dD2dt = v*((1 / (1 + (N2/KD)**h)) - D2)
    dD3dt = v*((1 / (1 + (N3/KD)**h)) - D3)
    return [dN1dt, dN2dt, dN3dt, dD1dt, dD2dt, dD3dt]


# In[26]:


def simulate_triangle(A, KD, t, state0, alpha=ALPHA, beta=1.0):
    """Symmetric triangle: all three contacts equal to A."""
    dt = t[1] - t[0]
    res = odeint(ThreeCellModel, state0, t, args=(A, A, A, KD), tfirst=True)
    N = [res[:, 0], res[:, 1], res[:, 2]]
    D = [res[:, 3], res[:, 4], res[:, 5]]

    idx = max(steady_state_idx(x) for x in N + D)   # all SIX variables
    T_ss = t[idx]

    R = [reporter(n, dt, alpha, beta) for n in N]
    rate = [(Rk[idx] - Rk[0]) / (T_ss - t[0]) for Rk in R]

    return dict(t=t, N=N, D=D, R=R, idx=idx, T_ss=T_ss, rate=rate,
                N_ss=[n[idx] for n in N], D_ss=[d[idx] for d in D],
                N_end=[n[-1] for n in N])          # <-- NEW: endpoint, matches 2-cell


# In[27]:


#%% Example of time dynamics of the three cell model (symmetric triangle)
A = 0.5                   # all three contacts equal; effective K_N = c/A = 2.0
KD = 0.1
t = np.arange(0.0, 500.0, 0.01)
state0 = [0.010, 0.0105, 0.011, 0.99, 0.99, 0.99]   # cell 1 lowest Notch

out = simulate_triangle(A, KD, t, state0)
print(f'T_ss = {out["T_ss"]:.1f}')
for k in range(3):
    print(f'  Cell {k+1}: N_ss={out["N_ss"][k]:.4f}  D_ss={out["D_ss"][k]:.4f}  '
          f'rate={out["rate"][k]:.4f}')

cut = out['idx'] + 1
cols = ['#FDB863', '#E66101', '#8E0152']  # 3 cell colors distinct from N and D
plt.figure()
for k in range(3):
    plt.plot(t[:cut], out['N'][k][:cut], '-',  color=cols[k], lw=2, label=f'N{k+1}')
    plt.plot(t[:cut], out['D'][k][:cut], '--', color=cols[k], lw=2, label=f'D{k+1}')
plt.axvline(out['T_ss'], color='gray', ls=':', lw=1, label=f'$T_{{ss}}$={out["T_ss"]:.1f}')
plt.xlabel("Simulation time"); plt.ylabel("Notch and Delta levels")
plt.ylim(0, 1); plt.legend(fontsize=7, ncol=2, frameon=False)
plt.tight_layout(); plt.show()


# In[29]:


#%% Panel B: three-cell phase plots (3 rows x 2 columns, square panels)
# Rows    = Delta level, Notch level, reporter accumulation rate
# Columns = Cell 1, Cells 2/3 (cells 2 and 3 always share a fate: max
#           |N2 - N3| across the grid is ~1e-3, so one column covers both)

# ---- grids ----
a = np.logspace(-1.3, 0.5, 101)   # 101 K_N values, log-spaced ~0.05 to ~3.16
b = np.logspace(-1.3, 0.5, 101)   # 101 K_D values, same range
t = np.arange(0.0, 500.0, 0.01)
state0 = [0.010, 0.0105, 0.011, 0.99, 0.99, 0.99]   # cell 1 lowest Notch

# ---- sweep ----
na, nb = len(a), len(b)
Nf = [np.zeros((na, nb)) for _ in range(3)]   # Nf[0] = Nf1, Nf[1] = Nf2, Nf[2] = Nf3
Df = [np.zeros((na, nb)) for _ in range(3)]
mNG = [np.zeros((na, nb)) for _ in range(3)]  # reporter accumulation rate per cell
FinalTime = np.zeros((na, nb))                # system-wide T_ss
Ndiff = np.zeros((na, nb))                    # spread of final Notch (max - min)
N23diff = np.zeros((na, nb))                  # convergence check

for i, kn in enumerate(a):
    A = c / kn                                # contact area from K_N
    for j, kd in enumerate(b):
        out = simulate_triangle(A, kd, t, state0)
        FinalTime[i, j] = out['T_ss']
        Ndiff[i, j] = max(out['N_end']) - min(out['N_end'])
        N23diff[i, j] = abs(out['N_end'][1] - out['N_end'][2])
        for k in range(3):
            Nf[k][i, j] = out['N_ss'][k]
            Df[k][i, j] = out['D_ss'][k]
            mNG[k][i, j] = out['rate'][k]

print(f'max |N2 - N3| across the grid: {N23diff.max():.2e}')
print(f'fraction above 1e-4: {(N23diff > 1e-4).mean():.2%}')

# ---- symmetry-breaking boundary line ----
As = a
Bs = b
KDval_N = np.zeros(len(a))
for i in range(len(a)):
    row = Ndiff[i, :]
    crossings = np.where((row[:-1] > 0.01) & (row[1:] <= 0.01))[0]
    if len(crossings) == 0:
        KDval_N[i] = b[-1] if row[-1] > 0.01 else np.nan   # nan = no boundary
    else:
        j = crossings[0]
        f = (row[j] - 0.01) / (row[j] - row[j + 1])
        logKD = np.log10(b[j]) + f * (np.log10(b[j + 1]) - np.log10(b[j]))
        KDval_N[i] = 10**logKD

# ---- plot ----
X, Y = np.meshgrid(a, b)

DeltaColor = ["white", "#AECBDA", "#8DB5CB", "#7AA8C2", "#6592AF", "#4C6C8E", "#374A71"]
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)
NotchColor = ['white', '#006400']
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)
SpeedColor = ["#000004", "#420A68", "#932667", "#DD513A", "#FCA50A", "#FCFFA4"]  # inferno
Speedcm = LinearSegmentedColormap.from_list("Speed", SpeedColor, N=100)

# levels are bounded in [0, 1] by construction; the rate is not, so it gets a
# shared data-derived scale (same convention as the two-cell figure)
vmin_shared = min(mNG[0].min(), mNG[1].min())
vmax_shared = max(mNG[0].max(), mNG[1].max())

ROWS = [('Delta level',                 [Df[0], Df[1]],   Deltacm, 0.0, 1.0),
        ('Notch$^{ICD}$ level',         [Nf[0], Nf[1]],   Notchcm, 0.0, 1.0),
        ('Notch Reporter Accumulation Rate',
                                        [mNG[0], mNG[1]], Speedcm,
                                        vmin_shared, vmax_shared)]
COLS = ['Cell 1', 'Cells 2 and 3']

fig = plt.figure(figsize=(11, 15))
for ri, (label, mats, cmap, vmn, vmx) in enumerate(ROWS):
    for ci in range(2):
        ax = plt.subplot(3, 2, 2 * ri + ci + 1)
        im = ax.pcolor(X, Y, mats[ci].T, cmap=cmap, shading='auto',
                       vmin=vmn, vmax=vmx, rasterized=True)
        plt.colorbar(im, ax=ax, orientation='vertical', label=label)
        ax.plot(As, KDval_N, color='gray', linestyle='dashed')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('K Notch Activation'); ax.set_ylabel('K Delta Inhibition')
        ax.set_title(COLS[ci])
        ax.set_box_aspect(1)

fig.suptitle('Three-cell symmetric triangle: steady-state levels and reporter '
             f'accumulation rate (α={ALPHA})')
plt.tight_layout()
plt.savefig(f'three_cell_phase_alpha{ALPHA}.svg', bbox_inches='tight')
plt.savefig(f'three_cell_phase_alpha{ALPHA}.png', dpi=300, bbox_inches='tight')
plt.show()


# In[22]:


#%% Panel C: population-level reporter accumulation rate vs K_D
# averaged over all sampled K_N; the population curve is the mean of all three
# cells, so cells 2 and 3 contribute twice as much as cell 1
pop3 = (mNG[0] + mNG[1] + mNG[2]).mean(axis=0) / 3

plt.figure(figsize=(5, 4))
plt.plot(b, mNG[0].mean(axis=0), color='#FDB863', lw=2.4, label='Cell 1')
plt.plot(b, mNG[1].mean(axis=0), color='#8E0152', lw=2.4, label='Cells 2 and 3')
plt.plot(b, pop3, color='black', lw=2.4, label='Population (all 3 cells)')
plt.xscale('log')
plt.ylim(bottom=0)
plt.xlabel('K Delta Inhibition')
plt.ylabel('Notch Reporter Accumulation Rate')
plt.title(f'Population-level accumulation rate vs $K_D$ (α={ALPHA})')
plt.legend(frameon=False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'three_cell_population_alpha{ALPHA}.svg', bbox_inches='tight')
plt.savefig(f'three_cell_population_alpha{ALPHA}.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




