#!/usr/bin/env python
# coding: utf-8

"""
Two-cell Notch-Delta lateral inhibition model: parameter sweep, 
phase plots, time-evolution and reporter-accumulation figures.
 
Original script by Saman (samantaba), 260719.py.
Edited by Miriam.
Refactored: single simulation function, one vectorized steady-state
definition, no re-simulation for the boundary line, and self-contained
supplement / time-evolution cells so nothing clobbers the main sweep.
 
The numerical results are intended to be identical to the previous
version; only the structure changed.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap

# Publication-quality text: editable font in SVG exports.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 0.5



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

def run_sweep(a, b, t, state0, dtype=np.float32):
    """Integrate the model over the full (a x b) grid.

    Returns N1, N2, D1, D2 (levels) and mN1, mN2 (cumulative Notch
    reporters, each anchored so mN[..., 0] == 0).

    Memory note: each array is len(a) * len(b) * len(t) floats. For the
    101 x 101 x 15000 main grid that is ~1.2 GB per array in float64.
    Pass dtype=np.float32 to halve it if RAM is tight (results shift only
    at ~1e-7, well below the analysis tolerances).
    """
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
            c1 = np.cumsum(res[:, 0]) * 0.01; c1 -= c1[0]   # ∫N dt, anchored to 0
            c2 = np.cumsum(res[:, 1]) * 0.01; c2 -= c2[0]
            mN1[i, j, :] = c1
            mN2[i, j, :] = c2
    return N1, N2, D1, D2, mN1, mN2



#%% Parameters and main simulation sweep
a = np.logspace(-1.3, 0.5, 101)   # 101 K_N values, log-spaced ~0.05 to ~3.16
b = np.logspace(-1.3, 0.5, 101)   # 101 K_D values, same range
t = np.arange(0.0, 500.0, 0.01)   # 0..500, step 0.01 (500,000 points)

# The cell that starts slightly higher becomes Notch-ON. The 0.01 / 0.011
# order makes cell 2 the Notch-ON cell, matching the figures.
state0 = [0.01, 0.011, 0.99, 0.99]

N1, N2, D1, D2, mN1, mN2 = run_sweep(a, b, t, state0)




#%% Steady-state index (anchored on cell 2 Notch, N2) + convergence check
# rel_tol=1e-5 removes the artifacts seen at looser tolerances (1e-2..1e-4).
FinalIndex, not_converged = final_index_grid(N2, rel_tol=1e-5)

if not_converged:
    examples = [(a[i], b[j]) for (i, j) in not_converged[:5]]
    print(f"WARNING: {len(not_converged)} of {a.size * b.size} combos did NOT "
          f"converge within t={t[-1]:.0f} at rel_tol=1e-5 "
          f"(using last index as steady state). Examples: {examples}")
else:
    print(f"All {a.size * b.size} combos converged within t={t[-1]:.0f}.")

# If you want a per-protein steady state instead of the shared N2 anchor:
#   FinalIndexN1, _ = final_index_grid(N1, 1e-5)
#   FinalIndexN2, _ = final_index_grid(N2, 1e-5)
#   FinalIndexD1, _ = final_index_grid(D1, 1e-5)
#   FinalIndexD2, _ = final_index_grid(D2, 1e-5)




#%% Average Notch speed from the accumulated reporter (mN)
mNG1 = avg_speed(mN1, FinalIndex, t)
mNG2 = avg_speed(mN2, FinalIndex, t)




#%% Final-level slices at each cell's steady-state index
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




#%% 1D average speed vs K_D (averaged over K_N)
mNG1avg = np.mean(mNG1, axis=0)   # one value per K_D
mNG2avg = np.mean(mNG2, axis=0)

label_mNG1 = 'Cell 1 ISC'   # SC tied to mNG1
label_mNG2 = 'Cell 2 EB'    # EB tied to mNG2

plt.close('all')
plt.figure()
plt.plot(b, mNG1avg, color='#ad92b3', label=label_mNG1)
plt.plot(b, mNG2avg, color='#8eb644', label=label_mNG2)
plt.plot(b, (mNG1avg + mNG2avg) / 2, color='#000000', label='Population')
plt.xscale('log')
plt.legend()
plt.ylabel("Average Speed")
plt.xlabel("K_D")
plt.tight_layout()
plt.savefig('population_avg_speed.png', dpi=300, bbox_inches='tight')
plt.savefig('population_avg_speed.svg', bbox_inches='tight')
plt.show()




#%% Delta final-level phase plots
X, Y = np.meshgrid(a, b)
DeltaColor = ["white", "#AECBDA", "#8DB5CB", "#7AA8C2", "#6592AF", "#4C6C8E", "#374A71"]
Deltacm = LinearSegmentedColormap.from_list("Custom", DeltaColor, N=100)

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, Df1.T, cmap=Deltacm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Delta Level')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, Df2.T, cmap=Deltacm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Delta Level')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')

plt.tight_layout()
plt.savefig('Delta_final_levels.png', dpi=300, bbox_inches='tight')
plt.savefig('Delta_final_levels.svg', bbox_inches='tight')
plt.show()




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

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, Nf2.T, cmap=Notchcm, shading='auto', vmin=0, vmax=1, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Notch Level')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')

plt.tight_layout()
plt.savefig('Notch_final_levels.png', dpi=300, bbox_inches='tight')
plt.savefig('Notch_final_levels.svg', bbox_inches='tight')
plt.show()



#%% Notch speed phase plots
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
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Average Speed')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, mNG2.T, cmap=Speedcm, shading='auto',
                 vmin=vmin_shared, vmax=vmax_shared, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Average Speed')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')

fig.suptitle('Average Notch Activation Speed from T=0 to Steady State')
plt.tight_layout()
#plt.savefig('avg_speed_0_to_SS.png', dpi=300, bbox_inches='tight')
plt.savefig('avg_speed_0_to_SS.svg', bbox_inches='tight')
plt.show()




#%% Notch speed phase plots, with full scale color bar
X, Y = np.meshgrid(a, b)
SpeedColor = ["#000000","#000004", "#420A68", "#932667", "#DD513A", "#FCA50A", "#FCFFA4","#FFFFFF"]  # inferno + black to white
Speedcm = LinearSegmentedColormap.from_list("Speed", SpeedColor, N=100)

fig = plt.figure(figsize=(11, 5))

ax1 = plt.subplot(121)
im1 = ax1.pcolor(X, Y, mNG1.T, cmap=Speedcm, shading='auto',
                 vmin=0, vmax=1, rasterized=True)
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Average Speed')
ax1.plot(As, KDval_N, color='gray', linestyle='dashed')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('K Notch Activation'); ax1.set_ylabel('K Delta Inhibition')
ax1.set_title('Cell 1')

ax2 = plt.subplot(122)
im2 = ax2.pcolor(X, Y, mNG2.T, cmap=Speedcm, shading='auto',
                 vmin=0, vmax=1, rasterized=True)
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Average Speed')
ax2.plot(As, KDval_N, color='gray', linestyle='dashed')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('K Notch Activation'); ax2.set_ylabel('K Delta Inhibition')
ax2.set_title('Cell 2')

fig.suptitle('Average Notch Activation Speed from T=0 to Steady State')
plt.tight_layout()
plt.savefig('avg_speed_0_to_SS_new.svg', bbox_inches='tight')
plt.show()




#%% Supplement: average speed vs K_D for each fixed K_N
# Self-contained: uses local names (sup_*) so it never overwrites the
# main-sweep arrays (a, b, t, N1, mNG1, ...) above.
sup_a = np.array([0.5]) # chosen K_N values. this can change 
#sup_a = np.logspace(-1.3, 0.5, 10)    # 10 sampled K_N values on log scale (can change)
sup_b = np.logspace(-1.3, 0.5, 101)
sup_t = np.arange(0.0, 500.0, 0.01)
sup_state0 = [0.01, 0.011, 0.99, 0.99]

_, sup_N2, _, _, sup_mN1, sup_mN2 = run_sweep(sup_a, sup_b, sup_t, sup_state0)
sup_FinalIndex, _ = final_index_grid(sup_N2, rel_tol=1e-5)
sup_mNG1 = avg_speed(sup_mN1, sup_FinalIndex, sup_t)
sup_mNG2 = avg_speed(sup_mN2, sup_FinalIndex, sup_t)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(sup_a)))

plt.figure()
for i in range(len(sup_a)):
    plt.plot(sup_b, sup_mNG1[i, :], color=colors[i], label=f'K_N={sup_a[i]:.3f}')
plt.xscale('log'); 
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(bottom=0)          # force the floor to 0, let the top autoscale
plt.ylabel("Speed"); plt.xlabel("K_D")
plt.title("Notch Speed vs K_D  Cell 1")
plt.tight_layout()
plt.savefig('speed_vs_KD_cell1.png', dpi=300, bbox_inches='tight')
plt.savefig('speed_vs_KD_cell1.svg', bbox_inches='tight')
plt.show()

plt.figure()
for i in range(len(sup_a)):
    plt.plot(sup_b, sup_mNG2[i, :], color=colors[i], label=f'K_N={sup_a[i]:.3f}')
plt.xscale('log'); 
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(bottom=0)          # force the floor to 0, let the top autoscale
plt.ylabel("Speed"); plt.xlabel("K_D")
plt.title("Notch Speed vs K_D  Cell 2")
plt.tight_layout()
plt.savefig('speed_vs_KD_cell2.png', dpi=300, bbox_inches='tight')
plt.savefig('speed_vs_KD_cell2.svg', bbox_inches='tight')
plt.show()



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




#%% N/D time-evolution for chosen (K_N, K_D) pairs
# All curves are plotted over a FIXED display window (0 to T_WINDOW).
# A small triangle is placed just ABOVE each curve where it reaches steady state.
# Steady state is detected from the FULL trajectory (out to te_t[-1]) so the
# detection is accurate; the marker is only drawn if that point falls inside
# the display window. Curves that settle after the window get no marker and
# are flagged as ">T_WINDOW" in the legend.

te_t = np.arange(0.0, 500.0, 0.01)   # integrate long enough for accurate steady state
te_state0 = [0.01, 0.011, 0.99, 0.99]
pairs = [(0.25, 0.1), (0.25, 0.25)]
# filename-safe tag from the tested pairs, e.g. "KN0p5_KD0p25__KN0p5_KD0p5"
pair_tag = '__'.join(f'KN{kn}_KD{kd}'.replace('.', 'p') for (kn, kd) in pairs)

T_WINDOW = 50.0                                  # fixed x-axis cutoff
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

# N2 steady-state time for each pair (the shared anchor used elsewhere)
n2_ss = []
for (kn, kd, n1, n2, d1, d2) in trajectories:
    idx = steady_state_idx(n2, rel_tol=1e-5)
    n2_ss.append((kd, te_t[idx], idx < cut))    # (K_D, t_ss, is it inside the window?)

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


#def draw_n2_ss_lines():
#    """Vertical dotted line at each pair's N2 steady-state time (if in-window)."""
#    for (kd, t_ss, in_win) in n2_ss:
#        if in_win:
#            plt.axvline(t_ss, color='gray', linestyle=':', lw=1, zorder=1,
#                        label=f'N2 steady state (K_D={kd}, t={t_ss:.1f})')

def draw_n2_ss_lines():
    """Vertical line at each pair's N2 steady-state time (if in-window).
    Solid for the lower K_D, dashed for the higher (matches curve linestyle)."""
    for (kd, t_ss, in_win) in n2_ss:
        if in_win:
            plt.axvline(t_ss, color='gray', linestyle=ls_for_kd(kd), lw=1, zorder=1,
                        label=f'N2 steady state (K_D={kd}, t={t_ss:.1f})')

plt.close('all');
# PASS 2: Cell 1
plt.figure(figsize=(4, 3))
for (kn, kd, n1, n2, d1, d2) in trajectories:
    ls = ls_for_kd(kd)
    plot_curve(n1, NOTCH_COLOR, ls, 'N1', kn, kd)
    plot_curve(d1, DELTA_COLOR, ls, 'D1', kn, kd)
draw_n2_ss_lines()
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
draw_n2_ss_lines()
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




#%% Notch reporter accumulation over a K_D series (choose K_N, K_D)
ra_t = np.arange(0.0, 500.0, 0.01)
ra_state0 = [0.01, 0.011, 0.99, 0.99]
ra_pairs = pairs
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
plt.xlim(0,50); plt.ylim(0,35)
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
plt.xlim(0,50); plt.ylim(0,35)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.savefig(f'cell2_reporter_{pair_tag}.svg', bbox_inches='tight')
plt.show()



