#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-cell Notch-Delta lateral inhibition model: parameter sweep, 
phase plots, time-evolution and reporter-accumulation figures.
 
Original script by Miriam, 20260721_Modeling_Code_MS.ipynb 
Refactored: single simulation function, one vectorized steady-state
definition, no re-simulation for the boundary line, and self-contained
supplement / time-evolution cells so nothing clobbers the main sweep.
 
Reporter now includes first-order degradation (alpha), 
changed steady-state definition to all 4 variables,
t extended to 500.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import lfilter
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Publication-quality text: editable font in SVG exports.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'     # keeps text editable in Illustrator
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'


# In[3]:


#%% Model definitions
r = h = 2      # Hill coefficients
v = 1.0        # Delta timescale

def GuisoniModel(t, state, a, b):
    """a = K_N (Notch activation threshold), b = K_D (Delta inhibition)."""
    N1, N2, D1, D2 = state
    dN1dt = (D2**r / (a**r + D2**r)) - N1
    dN2dt = (D1**r / (a**r + D1**r)) - N2
    dD1dt = v * ((1 / (1 + (N1 / b)**h)) - D1)
    dD2dt = v * ((1 / (1 + (N2 / b)**h)) - D2)
    return [dN1dt, dN2dt, dD1dt, dD2dt]


# In[4]:


#%% Helper functions (one definition of each, used everywhere below)

#relative tolerance choice
REL_TOL = 1e-5 

def steady_state_idx(arr, rel_tol=REL_TOL):
    """Earliest index after which `arr` never again strays more than
    rel_tol (relative to its final value) from that final value.

    Vectorized O(n) version of the old loop-based method: the answer is
    just (last index that is still out of tolerance) + 1.
    """
    final_val = arr[-1]
    threshold = rel_tol * max(abs(final_val), 1e-8)
    beyond = np.where(np.abs(arr - final_val) >= threshold)[0]
    if beyond.size == 0:          # already within tol for the whole trace
        return 0
    return min(int(beyond[-1]) + 1, len(arr) - 1)   # fallback: last index

def final_index_grid(arr3d, rel_tol=REL_TOL):
    """Steady-state index for every (i, j) cell of a (na, nb, nt) array.

    Returns the integer index grid and a list of (i, j) pairs that never
    settled within the time window (they fall back to the last index).
    """
    na, nb, nt = arr3d.shape
    fi = np.zeros((na, nb), dtype=int)
    not_converged = []
    for i in range(na):
        for j in range(nb):
            idx = steady_state_idx(arr3d[i, j, :], rel_tol)
            fi[i, j] = idx
            if idx == nt - 1:
                not_converged.append((i, j))
    return fi, not_converged


def reporter(N, dt, alpha, beta=1.0):
    """Solve dR/dt = beta*N - alpha*R, R(0)=0, holding N constant over each step.
    The alpha=0 branch is the alpha->0 limit of the alpha>0 branch
    (left-Riemann sum of N), so results are continuous in alpha."""
    if alpha == 0:
        R = np.concatenate([[0.0], np.cumsum(N[:-1]) * dt * beta])
        return R
    decay = np.exp(-alpha * dt)
    return lfilter([0.0, 1.0], [1.0, -decay], beta * N * (1.0 - decay) / alpha)


# actually more accurate as "mean_accumulated_rate" but didn't want to change future cell calls
def avg_speed(mN, final_index, t):
    """Average reporter accumulation rate from t0 to each cell's steady state."""
    na, nb = final_index.shape
    out = np.zeros((na, nb))
    ti = t[0]
    for i in range(na):
        for j in range(nb):
            idx = int(final_index[i, j])
            out[i, j] = (mN[i, j, idx] - mN[i, j, 0]) / (t[idx] - ti)
    return out

# def cross_time(mN, t, thresh):
#     """First time each grid cell's reporter reaches thresh.
#     np.nan where it never does (cell never becomes NRE-hi)."""
#     na, nb, _ = mN.shape
#     out = np.full((na, nb), np.nan)
#     for i in range(na):
#         for j in range(nb):
#             k = np.where(mN[i, j, :] >= thresh)[0]
#             if k.size:
#                 out[i, j] = t[k[0]]
#     return out


def run_sweep(a, b, t, state0, alpha, dtype=np.float32):   # CHANGED: alpha added
    """Integrate the model over the full (a x b) grid.

    Returns N1, N2, D1, D2 (levels) and mN1, mN2 (Notch reporter levels,
    each anchored so mN[..., 0] == 0).

    The reporter obeys dR/dt = beta*N - alpha*R with beta = 1 and R(0) = 0.
    Passing alpha=0 reproduces the previous cumulative-integral behaviour.

    Memory note: each array is len(a) * len(b) * len(t) floats. For the
    101 x 101 x 50000 main grid that is ~4 GB per array in float64.
    Pass dtype=np.float32 to halve it if RAM is tight (results shift only
    at ~1e-7, well below the analysis tolerances).
    """
    dt = t[1] - t[0]                                        # CHANGED: was 0.01
    shape = (len(a), len(b), len(t))
    N1 = np.zeros(shape, dtype=dtype)
    N2 = np.zeros(shape, dtype=dtype)
    D1 = np.zeros(shape, dtype=dtype)
    D2 = np.zeros(shape, dtype=dtype)
    mN1 = np.zeros(shape, dtype=dtype)
    mN2 = np.zeros(shape, dtype=dtype)
    for i in range(len(a)):
        for j in range(len(b)):
            res = odeint(GuisoniModel, state0, t, args=(a[i], b[j]), tfirst=True)
            N1[i, j, :] = res[:, 0]
            N2[i, j, :] = res[:, 1]
            D1[i, j, :] = res[:, 2]
            D2[i, j, :] = res[:, 3]
            c1 = reporter(res[:, 0], dt, alpha)             # CHANGED: was cumsum
            c2 = reporter(res[:, 1], dt, alpha)             # CHANGED: was cumsum
            mN1[i, j, :] = c1
            mN2[i, j, :] = c2
    return N1, N2, D1, D2, mN1, mN2


# In[5]:


#%% Parameters and main simulation sweep
a = np.logspace(-1.3, 0.5, 101)   # 101 K_N values, log-spaced ~0.05 to ~3.16
b = np.logspace(-1.3, 0.5, 101)   # 101 K_D values, same range
t = np.arange(0.0, 500.0, 0.01)   # 0..500, step 0.01 (50,000 points)

# The cell that starts slightly higher becomes Notch-ON. The 0.01 / 0.011
# order makes cell 2 the Notch-ON cell, matching the figures.
state0 = [0.01, 0.011, 0.99, 0.99]

ALPHA = 0.1            # estimate on tau_half(active Notch) / tau_half(reporter). sweep 0.01-1 for robustness

N_THR = 0.1          # Notch-positive threshold; 0.1 = Guisoni et al. 2017
MODE = 'fate'        # 'fate'      : classify each cell by N > N_THR (Guisoni et al. 2017)
                     # 'symmetry'  : asymmetric = |N1-N2| > SYM_TOL (matches dashed line);
                     #               symmetric pairs split by N > N_THR
SYM_TOL = 0.01       # same cutoff as the symmetry-breaking boundary line

N1, N2, D1, D2, mN1, mN2 = run_sweep(a, b, t, state0, alpha=ALPHA)


# In[6]:


#%% Steady-state index + convergence check
# rel_tol=1e-5 removes the artifacts seen at looser tolerances (1e-2..1e-4).
#FinalIndex, not_converged = final_index_grid(N2, rel_tol=1e-5)

#%% Steady-state index: earliest index after which all 4 state variables
#   stay within rel_tol of their final values
results = [final_index_grid(X, rel_tol=REL_TOL) for X in (N1, N2, D1, D2)]
FinalIndex = np.maximum.reduce([fi for fi, _ in results])
not_converged = sorted(set().union(*[set(nc) for _, nc in results]))

if not_converged:
    examples = [(a[i], b[j]) for (i, j) in not_converged[:5]]
    print(f"WARNING: {len(not_converged)} of {a.size * b.size} combos did NOT "
          f"converge within t={t[-1]:.0f} at rel_tol={REL_TOL} "
          f"(using last index as steady state). Examples: {examples}")
else:
    print(f"All {a.size * b.size} combos converged within t={t[-1]:.0f}.")

#per-protein steady state instead of the shared anchor:
#   FinalIndexN1, _ = final_index_grid(N1, 1e-5)
#   FinalIndexN2, _ = final_index_grid(N2, 1e-5)
#   FinalIndexD1, _ = final_index_grid(D1, 1e-5)
#   FinalIndexD2, _ = final_index_grid(D2, 1e-5)


# In[7]:


#%% Mean rate of Notch reporter accumulation (reporter with degradation, α = ALPHA)
mNG1 = avg_speed(mN1, FinalIndex, t)
mNG2 = avg_speed(mN2, FinalIndex, t)

# Notch itself (reporter-free): net change in N over [0, T_ss], divided by T_ss
NS1 = avg_speed(N1, FinalIndex, t)
NS2 = avg_speed(N2, FinalIndex, t)


# In[8]:


#%% Final-level slices at each grid point's steady-state index
na, nb = len(a), len(b)
Nf1 = np.zeros((na, nb)); Nf2 = np.zeros((na, nb))
Df1 = np.zeros((na, nb)); Df2 = np.zeros((na, nb))
for i in range(na):
    for j in range(nb):
        idx = int(FinalIndex[i, j])
        Nf1[i, j] = N1[i, j, idx]
        Nf2[i, j] = N2[i, j, idx]
        Df1[i, j] = D1[i, j, idx]
        Df2[i, j] = D2[i, j, idx]


# In[9]:


#%% Patch: re-run combos that hadn't converged by t[-1] (near the boundary,
#   critical slowing down) on a longer window, and overwrite their values
T_LONG = 3000.0
dt = t[1] - t[0]
t_long = np.arange(0.0, T_LONG, dt)
patched_tss = {}
still_bad = []

for (i, j) in not_converged:
    res = odeint(GuisoniModel, state0, t_long, args=(a[i], b[j]), tfirst=True)
    idx = max(steady_state_idx(res[:, k]) for k in range(4))
    if idx == len(t_long) - 1:
        still_bad.append((i, j))
        continue
    R1 = reporter(res[:, 0], dt, ALPHA)
    R2 = reporter(res[:, 1], dt, ALPHA)
    tss = t_long[idx]
    mNG1[i, j] = (R1[idx] - R1[0]) / tss
    mNG2[i, j] = (R2[idx] - R2[0]) / tss
    NS1[i, j] = (res[idx, 0] - res[0, 0]) / tss      # Notch Speed
    NS2[i, j] = (res[idx, 1] - res[0, 1]) / tss      # Notch Speed
    Nf1[i, j], Nf2[i, j], Df1[i, j], Df2[i, j] = res[idx, :4]
    patched_tss[(a[i], b[j])] = tss

print(f"Patched {len(patched_tss)} combos; t_ss range "
      f"{min(patched_tss.values()):.0f}-{max(patched_tss.values()):.0f}")
if still_bad:
    print(f"Still not converged by t={T_LONG:.0f}: {still_bad}")


# In[10]:


#%% Symmetry-breaking boundary line (homogeneous vs. symmetry-broken)
As = a                       # boundary is evaluated on the main grid
Bs = b

Ndiff = np.abs(N1[:, :, -1] - N2[:, :, -1])   # |N1 - N2| at final time
KDval_N = np.zeros(len(a))
#with Claude-suggested smoothing of the line

for i in range(len(a)):
    row = Ndiff[i, :]
    crossings = np.where((row[:-1] > 0.01) & (row[1:] <= 0.01))[0]
    if len(crossings) == 0:
        KDval_N[i] = b[-1] if row[-1] > 0.01 else np.nan   # nan = no boundary
    else:
        j = crossings[0]
        f = (row[j] - 0.01) / (row[j] - row[j + 1])        # interpolate to Ndiff == 0.01
        logKD = np.log10(b[j]) + f * (np.log10(b[j + 1]) - np.log10(b[j]))
        KDval_N[i] = 10**logKD

if MODE == 'fate':
    renewal = ((Nf1 <= N_THR) & (Nf2 <= N_THR)).astype(float)
else:
    renewal = ((Ndiff <= 0.01) & (np.maximum(Nf1, Nf2) <= N_THR)).astype(float)
RENEWAL_EDGE = '#4B2E83'
plt.rcParams['hatch.color'] = RENEWAL_EDGE


# In[11]:


#%% Fate-regime phase plot: asymmetric / symmetric terminal / symmetric renewal
# Uses Nf1, Nf2 (final levels, after the patch cell), Ndiff and KDval_N (boundary cell).

if MODE == 'fate':
    hi1, hi2 = Nf1 > N_THR, Nf2 > N_THR
    regime = np.where(hi1 & hi2, 1, np.where(~hi1 & ~hi2, 2, 0))
else:
    sym = Ndiff <= SYM_TOL
    regime = np.where(~sym, 0, np.where(np.maximum(Nf1, Nf2) > N_THR, 1, 2))
# 0 = asymmetric, 1 = symmetric terminal, 2 = symmetric renewal
renewal = (regime == 2).astype(float)

for k, name in enumerate(['Asymmetric', 'Symmetric terminal', 'Symmetric renewal']):
    print(f"{name}: {np.mean(regime == k):.1%} of grid")

REGIME_COLORS = ['#F28E2B', '#F6D55C', '#B39DDB']   # orange, yellow, purple
RENEWAL_EDGE = '#4B2E83'
plt.rcParams['hatch.color'] = RENEWAL_EDGE
X, Y = np.meshgrid(a, b)

fig, ax = plt.subplots(figsize=(6.2, 5))
ax.pcolormesh(X, Y, regime.T, cmap=ListedColormap(REGIME_COLORS),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), shading='auto', rasterized=True)
ax.contourf(X, Y, renewal.T, levels=[0.5, 1.5], colors='none', hatches=['////'])
ax.contour(X, Y, renewal.T, levels=[0.5], colors=RENEWAL_EDGE, linestyles='dotted', linewidths=1.6)
ax.plot(As, KDval_N, color='0.3', linestyle='dashed', lw=1.2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_box_aspect(1)
ax.set_xlabel('K Notch Activation'); ax.set_ylabel('K Delta Inhibition')
ax.set_title(f'Fate regimes ($N_{{thr}}$ = {N_THR})')
ax.legend(handles=[Patch(fc=REGIME_COLORS[0], label='Asymmetric'),
                   Patch(fc=REGIME_COLORS[1], label='Symmetric terminal'),
                   Patch(fc=REGIME_COLORS[2], hatch='////', ec=RENEWAL_EDGE, ls=':',
                         label='Symmetric renewal'),
                   plt.Line2D([], [], color='0.3', ls='--', label='Symmetry-breaking boundary')],
          loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f'fate_regimes_{MODE}_Nthr{N_THR}.svg', bbox_inches='tight')
plt.show()


# In[12]:


#%% Delta final-level phase plots
X, Y = np.meshgrid(a, b)
DeltaColor = ["white", "#AECBDA", "#8DB5CB", "#7AA8C2", "#6592AF", "#4C6C8E", "#374A71"]
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, Df1.T, cmap=Deltacm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Delta Level')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
# ax1.contour(X, Y, renewal.T, levels=[0.5], colors=RENEWAL_EDGE,
#             linestyles='dotted', linewidths=1.2)
# ax1.contourf(X, Y, renewal.T, levels=[0.5, 1.5], colors='none', hatches=['////'])
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')
ax1.set_box_aspect(1)

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, Df2.T, cmap=Deltacm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Delta Level')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
# ax2.contour(X, Y, renewal.T, levels=[0.5], colors=RENEWAL_EDGE,
#             linestyles='dotted', linewidths=1.2)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')
ax2.set_box_aspect(1)


plt.tight_layout()
plt.savefig('Delta_final_levels.png', dpi=300, bbox_inches='tight')
plt.savefig('Delta_final_levels.svg', bbox_inches='tight')
plt.show()


# In[13]:


#%% Notch final-level phase plots
X, Y = np.meshgrid(a, b)
NotchColor = ['white', '#006400']
Notchcm = LinearSegmentedColormap.from_list("Custom", NotchColor, N=100)

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, Nf1.T, cmap=Notchcm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Notch Level')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')
ax1.set_box_aspect(1)

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, Nf2.T, cmap=Notchcm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Notch Level')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')
ax2.set_box_aspect(1)

plt.tight_layout()
plt.savefig('Notch_final_levels.png', dpi=300, bbox_inches='tight')
plt.savefig('Notch_final_levels.svg', bbox_inches='tight')
plt.show()


# In[14]:


#%% Notch reporter accumulation phase plots
X, Y = np.meshgrid(a, b)
# SpeedColor = ["#4B1D69", "#8A5A9E", "#B9AFC0", "#D3D88A", "#CDEB3A"]  # yellow-purple
SpeedColor = ["#000004", "#420A68", "#932667", "#DD513A", "#FCA50A", "#FCFFA4"]  # inferno
# SpeedColor = ["#2D004B", "#7B2382", "#C13B7A", "#F06043", "#FDAE3B", "#FFF7B0"]  # purple-red-yellow
# SpeedColor = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]  # viridis
Speedcm = LinearSegmentedColormap.from_list("Speed", SpeedColor, N=100)

# shared color scale so both panels are directly comparable by eye
vmin_shared = min(mNG1.min(), mNG2.min())
vmax_shared = max(mNG1.max(), mNG2.max())

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, mNG1.T, cmap=Speedcm, shading='auto',
                 vmin=vmin_shared, vmax=vmax_shared, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Notch Reporter Accumulation Rate')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')
ax1.set_box_aspect(1)

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, mNG2.T, cmap=Speedcm, shading='auto',
                 vmin=vmin_shared, vmax=vmax_shared, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Notch Reporter Accumulation Rate')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')
ax2.set_box_aspect(1)

fig.suptitle('Mean rate of Notch reporter accumulation from T=0 to Steady State')
plt.tight_layout()
#plt.savefig('avg_speed_0_to_SS.png', dpi=300, bbox_inches='tight')
plt.savefig(f'reporter_rate_0_to_SS_alpha{ALPHA}.svg', bbox_inches='tight')
plt.show()


# In[15]:


#%% Notch speed/activation rate phase plots (reporter-free)
X, Y = np.meshgrid(a, b)
SpeedColor = ["#000004", "#420A68", "#932667", "#DD513A", "#FCA50A", "#FCFFA4"]  # inferno
Speedcm = LinearSegmentedColormap.from_list("Speed", SpeedColor, N=100)

# Cell 1's net change is slightly negative (~ -3e-4) where it rises and falls back;
# floor at 0 so those pixels render black rather than stretching the scale
NS_vmin = 0.0
NS_vmax = max(NS1.max(), NS2.max())

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, np.clip(NS1, 0, None).T, cmap=Speedcm, shading='auto',
                 vmin=NS_vmin, vmax=NS_vmax, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Mean rate of Notch activation')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')
ax1.set_box_aspect(1)

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, np.clip(NS2, 0, None).T, cmap=Speedcm, shading='auto',
                 vmin=NS_vmin, vmax=NS_vmax, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Mean rate of Notch activation')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')
ax2.set_box_aspect(1)

fig.suptitle('Mean rate of Notch activation from T=0 to Steady State')
plt.tight_layout()
plt.savefig('notch_activation_rate_0_to_SS.svg', bbox_inches='tight')
plt.show()


# In[19]:


#%% 1D mean rate of Notch reporter accumulation vs K_D (averaged over K_N)


#%% 1D mean rate of Notch reporter accumulation vs K_D (averaged over K_N)
EXCLUDE_RENEWAL = True
ren = renewal.astype(bool) if EXCLUDE_RENEWAL else np.zeros_like(mNG1, dtype=bool)

mNG1avg = np.nanmean(np.where(ren, np.nan, mNG1), axis=0)   # one value per K_D
mNG2avg = np.nanmean(np.where(ren, np.nan, mNG2), axis=0)
popavg  = (mNG1avg + mNG2avg) / 2   # unweighted mean of the pair (1:1 SC:EB)
print(f"Excluded {ren.sum()} symmetric-renewal points from the K_N average")

#%%old without exclusion
# mNG1avg = np.mean(mNG1, axis=0)   # one value per K_D
# mNG2avg = np.mean(mNG2, axis=0)
# popavg  = (mNG1avg + mNG2avg) / 2 # unweighted mean of the pair (1:1 SC:EB)

label_mNG1 = 'Cell 1'   # SC tied to mNG1
label_mNG2 = 'Cell 2'    # EB tied to mNG2

plt.close('all')
plt.figure()
plt.plot(b, mNG1avg, color='#ad92b3', label=label_mNG1)
plt.plot(b, mNG2avg, color='#8eb644', label=label_mNG2)
plt.plot(b, popavg,  color='#000000', label='Population')
plt.xscale('log')
plt.legend()
plt.ylabel("Mean rate of Notch reporter\naccumulation")
plt.xlabel("$K_D$")
plt.title(f"Averaged over $K_N$, α = {ALPHA}")
plt.tight_layout()
#plt.savefig(f'population_reporter_rate_alpha{ALPHA}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'population_reporter_rate_alpha{ALPHA}.svg', bbox_inches='tight')
plt.show()


print("Renewal points excluded:", int(ren.sum()))
print("K_D values with no data:", int(np.isnan(popavg).sum()))     # should be 0
popavg_all = (mNG1.mean(axis=0) + mNG2.mean(axis=0)) / 2
print("Max change from excluding renewal:", float(np.max(np.abs(popavg - popavg_all))))


# In[26]:


#%% average speed vs K_D for each fixed K_N
# Self-contained: uses local names (sup_*) so it never overwrites the
# main-sweep arrays (a, b, t, N1, mNG1, ...) above.
#sup_a = np.array([0.5]) # chosen K_N values. this can change 
sup_a = np.logspace(-1.3, 0.5, 10)    # 10 sampled K_N values on log scale (can change)
sup_b = np.logspace(-1.3, 0.5, 101)
sup_t = np.arange(0.0, 500.0, 0.01)
sup_state0 = [0.01, 0.011, 0.99, 0.99]

# Changes
sup_N1, sup_N2, sup_D1, sup_D2, sup_mN1, sup_mN2 = run_sweep(
    sup_a, sup_b, sup_t, sup_state0, alpha=ALPHA)

sup_FinalIndex = np.maximum.reduce([
    final_index_grid(X, rel_tol=1e-5)[0]
    for X in (sup_N1, sup_N2, sup_D1, sup_D2)
])

sup_mNG1 = avg_speed(sup_mN1, sup_FinalIndex, sup_t)
sup_mNG2 = avg_speed(sup_mN2, sup_FinalIndex, sup_t)

sup_mNG1_raw = sup_mNG1.copy()
sup_mNG2_raw = sup_mNG2.copy()
sup_mNG1 = np.where(sup_renewal, np.nan, sup_mNG1)
sup_mNG2 = np.where(sup_renewal, np.nan, sup_mNG2)

# Remove symmetric-renewal points (same rule as the fate-regime panel)
ii, jj = np.indices(sup_FinalIndex.shape)
sup_Nf1 = sup_N1[ii, jj, sup_FinalIndex]          # steady-state Notch, cell 1
sup_Nf2 = sup_N2[ii, jj, sup_FinalIndex]          # steady-state Notch, cell 2
sup_Ndiff = np.abs(sup_N1[:, :, -1] - sup_N2[:, :, -1])

if MODE == 'fate':
    sup_renewal = (sup_Nf1 <= N_THR) & (sup_Nf2 <= N_THR)
else:
    sup_renewal = (sup_Ndiff <= 0.01) & (np.maximum(sup_Nf1, sup_Nf2) <= N_THR)

sup_mNG1 = np.where(sup_renewal, np.nan, sup_mNG1)
sup_mNG2 = np.where(sup_renewal, np.nan, sup_mNG2)
print(f"Removed {sup_renewal.sum()} of {sup_renewal.size} points as symmetric renewal")
colors = plt.cm.viridis(np.linspace(0, 0.9, len(sup_a)))



plt.figure()
for i in range(len(sup_a)):
    plt.plot(sup_b, sup_mNG1[i, :], color=colors[i], label=f'K_N={sup_a[i]:.3f}')
plt.xscale('log'); 
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(bottom=0)          # force the floor to 0, let the top autoscale
plt.ylabel("Reporter accumulation rate"); plt.xlabel("K_D")
plt.title("Notch Reporter vs K_D  Cell 1")
plt.tight_layout()
plt.savefig('reporter_vs_KD_cell1.png', dpi=300, bbox_inches='tight')
plt.savefig('reporter_vs_KD_cell1.svg', bbox_inches='tight')
plt.show()

plt.figure()
for i in range(len(sup_a)):
    plt.plot(sup_b, sup_mNG2[i, :], color=colors[i], label=f'K_N={sup_a[i]:.3f}')
plt.xscale('log'); 
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(bottom=0)          # force the floor to 0, let the top autoscale
plt.ylabel("Reporter accumulation rate"); plt.xlabel("K_D")
plt.title("Notch Reporter vs K_D  Cell 2")
plt.tight_layout()
plt.savefig('reporter_vs_KD_cell2.png', dpi=300, bbox_inches='tight')
plt.savefig('reporter_vs_KD_cell2.svg', bbox_inches='tight')
plt.show()


# In[27]:


# plotting symmetric renewal lines as extended dotted lines
dot_mask = sup_renewal.copy()
dot_mask[:, 1:]  |= sup_renewal[:, :-1]
dot_mask[:, :-1] |= sup_renewal[:, 1:]

colors = plt.cm.viridis(np.linspace(0, 0.9, len(sup_a)))

def plot_cell(Z, title, fname):
    plt.figure()
    for i in range(len(sup_a)):
        has_solid = (~sup_renewal[i]).any()
        plt.plot(sup_b, np.where(sup_renewal[i], np.nan, Z[i]), color=colors[i],
                 label=f'K_N={sup_a[i]:.3f}' if has_solid else '_nolegend_')
        plt.plot(sup_b, np.where(dot_mask[i], Z[i], np.nan), color=colors[i], linestyle=':',
                 label='_nolegend_' if has_solid else f'K_N={sup_a[i]:.3f} (renewal)')
    plt.plot([], [], color='0.4', linestyle=':', label='Symmetric renewal')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim(bottom=0)
    plt.ylabel("Reporter accumulation rate"); plt.xlabel("K_D")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{fname}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{fname}.svg', bbox_inches='tight')
    plt.show()

plot_cell(sup_mNG1_raw, "Notch Reporter vs K_D  Cell 1", 'reporter_vs_KD_cell1')
plot_cell(sup_mNG2_raw, "Notch Reporter vs K_D  Cell 2", 'reporter_vs_KD_cell2')


# In[28]:


#%% N/D time-evolution for chosen (K_N, K_D) pairs
# Both cells share ONE global steady-state cutoff so the x-axes match.
te_t = np.arange(0.0, 500.0, 0.01)
te_state0 = [0.01, 0.011, 0.99, 0.99]
pairs = [(0.5, 0.25), (0.5, 1.0)]

NOTCH_COLOR = '#006400'
DELTA_COLOR = '#00008B'

# PASS 1: integrate every pair, find the global steady-state cutoff
trajectories = []
tf_global = 0
for (kn, kd) in pairs:
    res = odeint(GuisoniModel, te_state0, te_t, args=(kn, kd), tfirst=True)
    n1, n2, d1, d2 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
    tf = max(steady_state_idx(n1), steady_state_idx(n2),
             steady_state_idx(d1), steady_state_idx(d2))
    tf_global = max(tf_global, tf)
    trajectories.append((kn, kd, n1, n2, d1, d2))

# linestyle by K_D: highest tested K_D dashed, others solid
kd_values = [kd for (_, kd) in pairs]
def ls_for_kd(kd):
    return '--' if kd == max(kd_values) else '-'

# PASS 2: Cell 1
plt.figure(figsize=(7, 5))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plt.plot(te_t[:tf_global], n1[:tf_global], color=NOTCH_COLOR, linestyle=ls, lw=2.5,
             label=f'N1 (K_N={kn}, K_D={kd})')
    plt.plot(te_t[:tf_global], d1[:tf_global], color=DELTA_COLOR, linestyle=ls, lw=2.5,
             label=f'D1 (K_N={kn}, K_D={kd})')
plt.xlabel('Time'); plt.ylabel('Level')
plt.title('Cell 1: Notch (N1) and Delta (D1) time evolution')
plt.ylim(0, 1); plt.legend(fontsize=11); plt.tight_layout()
plt.savefig('N1_D1_time_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('N1_D1_time_evolution.svg', bbox_inches='tight')
plt.show()

# PASS 2: Cell 2
plt.figure(figsize=(7, 5))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plt.plot(te_t[:tf_global], n2[:tf_global], color=NOTCH_COLOR, linestyle=ls, lw=2.5,
             label=f'N2 (K_N={kn}, K_D={kd})')
    plt.plot(te_t[:tf_global], d2[:tf_global], color=DELTA_COLOR, linestyle=ls, lw=2.5,
             label=f'D2 (K_N={kn}, K_D={kd})')
plt.xlabel('Time'); plt.ylabel('Level')
plt.title('Cell 2: Notch (N2) and Delta (D2) time evolution')
plt.ylim(0, 1); plt.legend(fontsize=11); plt.tight_layout()
plt.savefig('N2_D2_time_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('N2_D2_time_evolution.svg', bbox_inches='tight')
plt.show()


# In[29]:


#%% N/D time-evolution for chosen (K_N, K_D) pairs
# All curves are plotted over a FIXED display window (0 to T_WINDOW).
# A marker is placed where each individual curve reaches steady state.
# Steady state is detected from the FULL trajectory (out to te_t[-1]) so the
# detection is accurate; the marker is only drawn if that point falls inside
# the display window. Curves that settle after the window get no marker and
# are flagged as ">T_WINDOW" in the legend.

te_t = np.arange(0.0, 500.0, 0.01)   # integrate long enough for accurate steady state
te_state0 = [0.01, 0.011, 0.99, 0.99]
pairs = [(0.5, 0.25), (0.5, 1.0)]

T_WINDOW = 100.0                                  # fixed x-axis cutoff
cut = int(np.searchsorted(te_t, T_WINDOW, side='right'))   # display index for t <= T_WINDOW

NOTCH_COLOR = '#006400'
DELTA_COLOR = '#374A71'

MARKER_OFFSET = 0.025    # how far above the line to float the triangle (data units)
MARKER_YMAX = 0.985      # clamp so markers near y=1 aren't clipped

# PASS 1: integrate every pair (full length, for accurate steady-state detection)
trajectories = []
for (kn, kd) in pairs:
    res = odeint(GuisoniModel, te_state0, te_t, args=(kn, kd), tfirst=True)
    trajectories.append((kn, kd, res[:, 0], res[:, 1], res[:, 2], res[:, 3]))

# linestyle by K_D: highest tested K_D dashed, others solid
kd_values = [kd for (_, kd) in pairs]
def ls_for_kd(kd):
    return '--' if kd == max(kd_values) else '-'

def plot_curve(arr, color, ls, name, kn, kd):
    """Plot one curve over the window and mark its steady state if in-window."""
    idx = steady_state_idx(arr, rel_tol=1e-5)   # index in the FULL trajectory
    t_ss = te_t[idx]
    if idx < cut:
        marker_y = min(arr[idx] + MARKER_OFFSET, MARKER_YMAX)
        # marker POINT: its own plot, linestyle='none', kept OUT of the legend
        plt.plot(te_t[idx], marker_y, marker='v', color=color, markersize=5,
                 markeredgecolor='black', markeredgewidth=0.5,
                 linestyle='none', zorder=5, label='_nolegend_')
        tag = f't_ss={t_ss:.1f}'
    else:
        tag = f't_ss>{T_WINDOW:.0f}'
    # curve LINE: pure line, NO marker argument, carries the label
    plt.plot(te_t[:cut], arr[:cut], color=color, linestyle=ls, lw=0.5,
             label=f'{name} (K_N={kn}, K_D={kd}, {tag})')

plt.close('all')

# PASS 2: Cell 1
plt.figure(figsize=(3, 2))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plot_curve(n1, NOTCH_COLOR, ls, 'N1', kn, kd)
    plot_curve(d1, DELTA_COLOR, ls, 'D1', kn, kd)
plt.xlim(0, T_WINDOW); plt.ylim(0, 1)
plt.xlabel('Relative time'); plt.ylabel('Level')
plt.title('Cell 1')
plt.plot([], [], marker='v', color='none', markeredgecolor='black',
         markeredgewidth=0.5, linestyle='none', label='steady state')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig('N1_D1_time_evolution.svg', bbox_inches='tight')
plt.show()

# PASS 2: Cell 2
plt.figure(figsize=(3, 2))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plot_curve(n2, NOTCH_COLOR, ls, 'N2', kn, kd)
    plot_curve(d2, DELTA_COLOR, ls, 'D2', kn, kd)
plt.xlim(0, T_WINDOW); plt.ylim(0, 1)
plt.xlabel('Relative time'); plt.ylabel('Level')
plt.title('Cell 2')
plt.plot([], [], marker='v', color='none', markeredgecolor='black',
         markeredgewidth=0.5, linestyle='none', label='steady state')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig('N2_D2_time_evolution.svg', bbox_inches='tight')
plt.show()


# In[56]:


#%% N/D time-evolution for chosen (K_N, K_D) pairs
# All curves are plotted over a FIXED display window (0 to T_WINDOW).
# A small triangle is placed just ABOVE each curve where it reaches steady state.
# Steady state is detected from the FULL trajectory (out to te_t[-1]) so the
# detection is accurate; the marker is only drawn if that point falls inside
# the display window. Curves that settle after the window get no marker and
# are flagged as ">T_WINDOW" in the legend.

te_t = np.arange(0.0, 500.0, 0.01)   # integrate long enough for accurate steady state
te_state0 = [0.01, 0.011, 0.99, 0.99]
pairs = [(0.75, 0.75), (0.75, 2.0)]
# filename-safe tag from the tested pairs, e.g. "KN0p5_KD0p25__KN0p5_KD0p5"
pair_tag = '__'.join(f'KN{kn}_KD{kd}'.replace('.', 'p') for (kn, kd) in pairs)

T_WINDOW = 55.0                                  # fixed x-axis cutoff
cut = int(np.searchsorted(te_t, T_WINDOW, side='right'))   # display index for t <= T_WINDOW

NOTCH_COLOR = '#006400'
DELTA_COLOR = '#00008B'

MARKER_OFFSET = 0.025    # how far above the line to float the triangle (data units)
MARKER_YMAX = 0.985      # clamp so markers near y=1 aren't clipped

# PASS 1: integrate every pair (full length, for accurate steady-state detection)
trajectories = []
for (kn, kd) in pairs:
    res = odeint(GuisoniModel, te_state0, te_t, args=(kn, kd), tfirst=True)
    trajectories.append((kn, kd, res[:, 0], res[:, 1], res[:, 2], res[:, 3]))

# System-wide steady-state time for each pair (the shared anchor used elsewhere):
# earliest time after which all four state variables stay within tolerance.
sys_ss = []
for (kn, kd, n1, n2, d1, d2) in trajectories:
    idx = max(steady_state_idx(x, rel_tol=1e-5) for x in (n1, n2, d1, d2))
    sys_ss.append((kd, te_t[idx], idx < cut))   # (K_D, t_ss, is it inside the window?)

# linestyle by K_D: highest tested K_D dashed, others solid
kd_values = [kd for (_, kd) in pairs]
def ls_for_kd(kd):
    return '--' if kd == max(kd_values) else '-'


def plot_curve(arr, color, ls, name, kn, kd):
    """Plot one curve over the window; mark its steady state with a triangle
    floated just above the line (only if the steady state is in-window)."""
    idx = steady_state_idx(arr, rel_tol=1e-5)   # index in the FULL trajectory
    t_ss = te_t[idx]
    if idx < cut:
        marker_y = min(arr[idx] + MARKER_OFFSET, MARKER_YMAX)   # sit above the line
        #plt.plot(te_t[idx], marker_y, marker='v', color=color, markersize=7,
                 #markeredgecolor='black', markeredgewidth=0.5,
                 #linestyle='none', zorder=5)
        tag = f't_ss={t_ss:.1f}'
    else:
        tag = f't_ss>{T_WINDOW:.0f}'          # settles after the display window
    plt.plot(te_t[:cut], arr[:cut], color=color, linestyle=ls, lw=1,
             label=f'{name} (K_N={kn}, K_D={kd}, {tag})')


#def draw_sys_ss_lines():
#    """Vertical dotted line at each pair's N2 steady-state time (if in-window)."""
#    for (kd, t_ss, in_win) in sys_ss:
#        if in_win:
#            plt.axvline(t_ss, color='gray', linestyle=':', lw=1, zorder=1,
#                        label=f'N2 steady state (K_D={kd}, t={t_ss:.1f})')

def draw_sys_ss_lines():
    """Vertical line at each pair's steady-state time (if in-window).
    Solid for the lower K_D, dashed for the higher (matches curve linestyle)."""
    for (kd, t_ss, in_win) in sys_ss:
        if in_win:
            plt.axvline(t_ss, color='gray', linestyle=ls_for_kd(kd), lw=1, zorder=1,
                        label=f'Steady state (K_D={kd}, t={t_ss:.1f})')

plt.close('all');
# PASS 2: Cell 1
plt.figure(figsize=(4, 3))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plot_curve(n1, NOTCH_COLOR, ls, 'N1', kn, kd)
    plot_curve(d1, DELTA_COLOR, ls, 'D1', kn, kd)
draw_sys_ss_lines()
plt.xlim(0, T_WINDOW); plt.ylim(0, 1)
plt.xlabel('Time'); plt.ylabel('Level')
plt.title('Cell 1: Notch (N1) and Delta (D1) time evolution')
#plt.plot([], [], marker='v', color='none', markeredgecolor='black',
         #markeredgewidth=0.8, linestyle='none', label='steady state')  # marker legend key
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig(f'N1_D1_time_evolution_{pair_tag}.svg', bbox_inches='tight')
plt.show()

# PASS 2: Cell 2
plt.figure(figsize=(4, 3))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plot_curve(n2, NOTCH_COLOR, ls, 'N2', kn, kd)
    plot_curve(d2, DELTA_COLOR, ls, 'D2', kn, kd)
draw_sys_ss_lines()
plt.xlim(0, T_WINDOW); plt.ylim(0, 1)
plt.xlabel('Time'); plt.ylabel('Level')
plt.title('Cell 2: Notch (N2) and Delta (D2) time evolution')
#plt.plot([], [], marker='v', color='none', markeredgecolor='black',
         #markeredgewidth=0.8, linestyle='none', label='steady state')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig(f'N2_D2_time_evolution_{pair_tag}.svg', bbox_inches='tight')
plt.show()


# In[60]:


#OLD DO NOT USE
#%% Notch reporter accumulation over a K_D series (choose K_N, K_D)
ra_t = np.arange(0.0, 500.0, 0.01)
ra_state0 = [0.01, 0.011, 0.99, 0.99]
ra_pairs = pairs
pairs = [(0.75, 0.75), (0.75, 2.0)]
# filename-safe tag from the tested pairs, e.g. "KN0p5_KD0p25__KN0p5_KD0p5"
pair_tag = '__'.join(f'KN{kn}_KD{kd}'.replace('.', 'p') for (kn, kd) in pairs)

ra_data = []
for (kn, kd) in ra_pairs:
    res = odeint(GuisoniModel, ra_state0, ra_t, args=(kn, kd), tfirst=True)
    n1, n2 = res[:, 0], res[:, 1]
    m1 = np.cumsum(n1) * 0.01; m1 -= m1[0]     # cell 1 reporter (∫N dt), anchored to 0
    m2 = np.cumsum(n2) * 0.01; m2 -= m2[0]     # cell 2 reporter (∫N dt), anchored to 0
    tf1 = steady_state_idx(n2)          # cell 1 truncated at N2 steady state (can change)
    tf2 = steady_state_idx(n2)          # cell 2 truncated at N2 steady state
    ra_data.append((kn, kd, m1, m2, tf1, tf2))

colors = plt.cm.magma(np.linspace(0, 0.9, len(ra_pairs)))

plt.figure(figsize=(4, 4))
for (kn, kd, m1, m2, tf1, tf2), color in zip(ra_data, colors):
    plt.plot(ra_t[:tf1], m1[:tf1], color=color, lw=2.5, label=f'K_N={kn}, K_D={kd}')
plt.xlabel('Time'); plt.ylabel('Notch reporter level')
plt.title('Cell 1: Notch reporter accumulation (t=0 to steady state)')
plt.xlim(0,55); plt.ylim(0,40)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig(f'cell1_reporter_{pair_tag}.svg', bbox_inches='tight')
plt.show()

plt.figure(figsize=(4, 4))
for (kn, kd, m1, m2, tf1, tf2), color in zip(ra_data, colors):
    plt.plot(ra_t[:tf2], m2[:tf2], color=color, lw=2.5, label=f'K_N={kn}, K_D={kd}')
plt.xlabel('Time'); plt.ylabel('Notch reporter level')
plt.title('Cell 2: Notch reporter accumulation (t=0 to steady state)')
plt.xlim(0,55); plt.ylim(0,40)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig(f'cell2_reporter_{pair_tag}.svg', bbox_inches='tight')
plt.show()


# In[61]:


#%% Notch reporter accumulation over a K_D series (choose K_N, K_D)
# All curves are plotted over a FIXED common window so their shapes are
# directly comparable. A triangle marks each condition's steady-state time
# (the end of the interval the accumulation rate is averaged over), and a
# thin dotted chord from the origin to that marker shows the mean
# accumulation rate itself as a slope.
ra_t = np.arange(0.0, 500.0, 0.01)
ra_dt = ra_t[1] - ra_t[0]
ra_state0 = [0.01, 0.011, 0.99, 0.99]
ra_pairs = pairs
pairs = [(0.75, 0.75), (0.75, 2.0)]
pair_tag = '__'.join(f'KN{kn}_KD{kd}'.replace('.', 'p') for (kn, kd) in pairs)

SHOW_CHORD = True        # straight line from origin to the T_ss marker
WINDOW_PAD = 1.15        # x-axis extends this far past the latest T_ss

ra_data = []
for (kn, kd) in ra_pairs:
    res = odeint(GuisoniModel, ra_state0, ra_t, args=(kn, kd), tfirst=True)
    n1, n2, d1, d2 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
    m1 = reporter(n1, ra_dt, ALPHA)        # dR/dt = N - alpha*R
    m2 = reporter(n2, ra_dt, ALPHA)
    tf = max(steady_state_idx(x) for x in (n1, n2, d1, d2))   # system-wide
    ra_data.append((kn, kd, m1, m2, tf))

# common display window, set by the latest steady state across all conditions
t_ss_max = max(ra_t[tf] for (_, _, _, _, tf) in ra_data)
T_WINDOW = t_ss_max * WINDOW_PAD
cut = int(np.searchsorted(ra_t, T_WINDOW, side='right'))

colors = plt.cm.magma(np.linspace(0, 0.9, len(ra_pairs)))


def plot_reporter_panel(which, fname_stem, title):
    """which = 1 or 2 (cell index). Plots all conditions on the common window."""
    plt.figure(figsize=(4, 4))
    ymax = 0.0
    for (kn, kd, m1, m2, tf), color in zip(ra_data, colors):
        m = m1 if which == 1 else m2
        t_ss = ra_t[tf]
        rate = (m[tf] - m[0]) / t_ss          # the metric: mean accumulation rate

        plt.plot(ra_t[:cut], m[:cut], color=color, lw=2.5,
                 label=f'K_N={kn}, K_D={kd} (rate={rate:.3f})')
        if SHOW_CHORD:
            plt.plot([0.0, t_ss], [m[0], m[tf]], color=color, lw=1.0,
                     linestyle=':', alpha=0.9, label='_nolegend_')
        plt.plot(t_ss, m[tf], marker='v', color=color, markersize=6,
                 markeredgecolor='black', markeredgewidth=0.5,
                 linestyle='none', zorder=5, label='_nolegend_')
        ymax = max(ymax, m[:cut].max())

    plt.xlabel('Time')
    plt.ylabel('Notch reporter level')
    plt.title(title)
    plt.xlim(0, T_WINDOW)
    plt.ylim(0, ymax * 1.08)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5),
               frameon=False, title='▼ = steady state\n⋯ = mean rate',
               title_fontsize=8)
    #plt.savefig(f'{fname_stem}_{pair_tag}.svg', bbox_inches='tight')
    plt.show()


plot_reporter_panel(1, 'cell1_reporter', 'Cell 1: Notch reporter accumulation')
plot_reporter_panel(2, 'cell2_reporter', 'Cell 2: Notch reporter accumulation')


# In[62]:


#%% Notch reporter accumulation over a K_D series (choose K_N, K_D)
# Each trace is truncated at its own system-wide steady-state time, which is
# the interval the accumulation rate is averaged over. All traces share one
# x-axis set by the latest steady state across conditions, so the differing
# durations are visible rather than hidden by per-panel rescaling.

ra_t = np.arange(0.0, 500.0, 0.01)
ra_dt = ra_t[1] - ra_t[0]
ra_state0 = [0.01, 0.011, 0.99, 0.99]
ra_pairs = pairs
pair_tag = '__'.join(f'KN{kn}_KD{kd}'.replace('.', 'p') for (kn, kd) in pairs)

AXIS_PAD = 1.05  # x-axis extends this far past the latest T_ss
Y_MAX = 10.0
X_MAX = 55.0

# # one common x-axis for every panel, set by the latest steady state
# t_ss_max = max(ra_t[tf] for (_, _, _, _, tf) in ra_data)
# X_MAX = t_ss_max * AXIS_PAD

ra_data = []
for (kn, kd) in ra_pairs:
    res = odeint(GuisoniModel, ra_state0, ra_t, args=(kn, kd), tfirst=True)
    n1, n2, d1, d2 = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
    m1 = reporter(n1, ra_dt, ALPHA)      # dR/dt = N - alpha*R
    m2 = reporter(n2, ra_dt, ALPHA)
    tf = max(steady_state_idx(x) for x in (n1, n2, d1, d2))   # system-wide
    ra_data.append((kn, kd, m1, m2, tf))

colors = plt.cm.magma(np.linspace(0, 0.9, len(ra_pairs)))

def plot_reporter_panel(which, fname_stem, title):
    """which = 1 or 2 (cell index). Each trace runs 0 -> its own T_ss."""
    plt.figure(figsize=(4, 4))
    for (kn, kd, m1, m2, tf), color in zip(ra_data, colors):
        m = m1 if which == 1 else m2
        t_ss = ra_t[tf]
        rate = (m[tf] - m[0]) / t_ss          # the metric

        plt.plot(ra_t[:tf + 1], m[:tf + 1], color=color, lw=2.5,
                 label=f'K_N={kn}, K_D={kd}\n  $T_{{ss}}$={t_ss:.1f}, rate={rate:.3f}')

    plt.xlabel('Time')
    plt.ylabel('Notch reporter level')
    plt.title(title)
    plt.xlim(0, X_MAX)
    plt.ylim(0, Y_MAX)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
               frameon=False)
    plt.savefig(f'{fname_stem}_{pair_tag}.svg', bbox_inches='tight')
    plt.show()


plot_reporter_panel(1, 'cell1_reporter', 'Cell 1: Notch reporter accumulation')
plot_reporter_panel(2, 'cell2_reporter', 'Cell 2: Notch reporter accumulation')


# In[63]:


#%% Alpha robustness sweep
# Re-runs the main grid at several reporter degradation rates and reports
# whether the K_D relationship holds throughout. Keyed by alpha so adding or
# removing a value means editing ALPHAS only.

ALPHAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]

C1, C2 = {}, {}                 # Cell 1 / Cell 2 accumulation-rate maps, per alpha
for al in ALPHAS:
    N1_a, N2_a, D1_a, D2_a, mN1_a, mN2_a = run_sweep(a, b, t, state0, alpha=al)
    FI_a = np.maximum.reduce([
        final_index_grid(X, rel_tol=1e-5)[0]
        for X in (N1_a, N2_a, D1_a, D2_a)
    ])
    C1[al] = avg_speed(mN1_a, FI_a, t)
    C2[al] = avg_speed(mN2_a, FI_a, t)
    print(f'alpha={al} done')

# ---- statistics ----
pop, S = {}, {}
for al in ALPHAS:
    pop[al] = (C1[al].mean(axis=0) + C2[al].mean(axis=0)) / 2
    S[al] = dict(popr=spearmanr(b, pop[al])[0])

print()
print('  alpha   population')
for al in ALPHAS:
    print(f'  {al:<8}{S[al]["popr"]:+.3f}')

# ---- figure ----
fig, ax = plt.subplots(figsize=(5.6, 4.2))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(ALPHAS)))

for al, c in zip(ALPHAS, cols):
    ax.plot(b, pop[al], color=c, lw=2.3,
            label=f'α = {al:<5}  ρ = {S[al]["popr"]:+.3f}')
ax.set_xscale('log')
ax.set_ylim(0, None)
ax.set_xlabel('$K_D$')
ax.set_ylabel('Population accumulation rate')
ax.set_title('Reporter accumulation rate increases with $K_D$\n'
             'at every reporter degradation rate', fontsize=11)
leg = ax.legend(fontsize=8.5, frameon=False, loc='upper left',
                title='Spearman ρ ($K_D$ vs rate)', title_fontsize=8.5)
leg._legend_box.align = 'left'
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('rebuttal_alpha_sweep.pdf', bbox_inches='tight')
plt.savefig('rebuttal_alpha_sweep.png', dpi=200, bbox_inches='tight')
plt.show()


# In[ ]:




