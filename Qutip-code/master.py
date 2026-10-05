import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ==========================================
# 1. System & Decoherence Parameters
# ==========================================
w_m = 2.0 * np.pi * 1000.0   # NiO Bus Frequency: 1.0 THz
w_q = 2.0 * np.pi * 980.0    # Qubit Frequency: 980 GHz (Dispersive: Detuning = 20 GHz)
g   = 2.0 * np.pi * 2.0      # Coupling Strength: 2.0 GHz

# Realistic Experimental Noise Rates
kappa_m  = 2.0 * np.pi * 0.050   # Bus Damping: 50 MHz (T_m ~ 20 ns)
gamma_t1 = 2.0 * np.pi * 0.002   # Qubit T1 relaxation: 2 MHz (T1 = 500 ns)
gamma_t2 = 2.0 * np.pi * 0.005   # Qubit T2* dephasing: 5 MHz (T2* = 200 ns)

N_m = 3  # Magnon Hilbert space truncation

# ==========================================
# 2. Operators & Hamiltonians
# ==========================================
a_m  = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(N_m))
sm_A = qt.tensor(qt.destroy(2), qt.qeye(2), qt.qeye(N_m))
sm_B = qt.tensor(qt.qeye(2), qt.destroy(2), qt.qeye(N_m))

H_q = 0.5 * w_q * (sm_A.dag() * sm_A - sm_A * sm_A.dag()) + \
      0.5 * w_q * (sm_B.dag() * sm_B - sm_B * sm_B.dag())
H_m = w_m * a_m.dag() * a_m
H_int = g * (sm_A.dag() * a_m + sm_A * a_m.dag()) + \
        g * (sm_B.dag() * a_m + sm_B * a_m.dag())
H = H_q + H_m + H_int

# Collapse operators for decoherence
c_ops = [
    np.sqrt(kappa_m) * a_m,
    np.sqrt(gamma_t1) * sm_A,
    np.sqrt(gamma_t1) * sm_B,
    np.sqrt(gamma_t2 / 2.0) * (sm_A.dag() * sm_A),
    np.sqrt(gamma_t2 / 2.0) * (sm_B.dag() * sm_B)
]

# Target Ideal Bell State: 1/sqrt(2) * (|e_A, g_B> - i|g_A, e_B>) x |0_m>
psi_target = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1), qt.basis(N_m, 0)) -
              1j * qt.tensor(qt.basis(2, 1), qt.basis(2, 0), qt.basis(N_m, 0))).unit()

# Initial State: |g_A, e_B, 0_m>
psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0), qt.basis(N_m, 0))
tlist = np.linspace(0, 2000e-3, 500) # 0 to 2000 ps (2 ns)

# ==========================================
# 3. Time Evolution (Ideal vs Open System)
# ==========================================
res_ideal = qt.mesolve(H, psi0, tlist, c_ops=[], e_ops=[])
res_open  = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[])

fidelities_ideal = []
fidelities_open  = []
infidelities_open = []
concurrence_ideal = []
concurrence_open  = []

for s_i, s_o in zip(res_ideal.states, res_open.states):
    # Fidelities
    F_i = qt.fidelity(s_i, psi_target)**2
    F_o = qt.fidelity(s_o, psi_target)**2
    fidelities_ideal.append(F_i)
    fidelities_open.append(F_o)
    infidelities_open.append(1.0 - F_o)

    # Concurrences
    concurrence_ideal.append(qt.concurrence(qt.ptrace(s_i, [0, 1])))
    concurrence_open.append(qt.concurrence(qt.ptrace(s_o, [0, 1])))

# ==========================================
# 4. Two-Panel Visualization
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Panel 1: Bell State Fidelity & Infidelity
ax1.plot(tlist * 1e3, fidelities_open, 'g-', lw=2.2, label=r'Open System Fidelity $\mathcal{F}(t)$')
ax1.plot(tlist * 1e3, fidelities_ideal, 'g--', lw=1.5, alpha=0.7, label=r'Ideal Fidelity $\mathcal{F}_{\mathrm{ideal}}(t)$')
ax1.plot(tlist * 1e3, infidelities_open, 'r-', lw=2, label=r'Gate Infidelity / Error $\epsilon(t) = 1 - \mathcal{F}$')
ax1.set_ylabel('Fidelity / Infidelity', fontsize=11)
ax1.set_title('Quantum Gate Fidelity and Infidelity under Decoherence', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='center right', fontsize=9)

# Panel 2: Concurrence Degradation
ax2.plot(tlist * 1e3, concurrence_open, color='darkorange', lw=2.5, label=r'Open System Concurrence $C_{AB}(t)$')
ax2.plot(tlist * 1e3, concurrence_ideal, color='darkorange', linestyle='--', lw=1.5, alpha=0.7, label=r'Ideal Concurrence $C_{AB}^{\mathrm{ideal}}(t)$')
ax2.axhline(1.0, color='red', linestyle=':', alpha=0.6, label='Maximally Entangled Bell State ($C_{AB}=1.0$)')
ax2.set_xlabel('Time (picoseconds)', fontsize=11)
ax2.set_ylabel('Concurrence $C_{AB}$', fontsize=11)
ax2.set_title('Entanglement Degradation ($T_1 = 500\,\mathrm{ns}, T_2^* = 200\,\mathrm{ns}, \kappa_m/2\pi = 50\,\mathrm{MHz}$)', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()