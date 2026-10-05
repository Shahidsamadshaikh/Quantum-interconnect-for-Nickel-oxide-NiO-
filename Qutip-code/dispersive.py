import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ==========================================
# 1. Dispersive System Parameters
# ==========================================
w_m = 2.0 * np.pi * 1000.0   # NiO Magnon Bus: 1.0 THz (1000 GHz)
w_q = 2.0 * np.pi * 980.0    # Qubits detuned by 20 GHz (Dispersive Regime)
g   = 2.0 * np.pi * 2.0      # Qubit-Bus Coupling Strength: 2.0 GHz

# Effective dispersive qubit-qubit coupling: g_eff = g^2 / Delta
delta = w_q - w_m            # Detuning = -20 GHz
g_eff = (g**2) / abs(delta)  # Effective virtual magnon swap rate

# Dissipative Parameters (Open System Dynamics)
kappa_m = 2.0 * np.pi * 0.050 # Magnon Damping (50 MHz)
gamma_q = 2.0 * np.pi * 0.010 # Qubit Relaxation (10 MHz)
n_th    = 0.0                 # Thermal Occupancy at THz scales (T = 15 mK)

# ==========================================
# 2. Operators & Tensor Hilbert Space Setup
# ==========================================
N_m = 3  # Magnon cavity truncation (virtual excitations stay near 0)

# Qubit A (0), Qubit B (1), Magnon Bus (2)
a_m  = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(N_m))
sm_A = qt.tensor(qt.destroy(2), qt.qeye(2), qt.qeye(N_m))
sm_B = qt.tensor(qt.qeye(2), qt.destroy(2), qt.qeye(N_m))

# Full System Hamiltonian
H_q = 0.5 * w_q * (sm_A.dag() * sm_A - sm_A * sm_A.dag()) + \
      0.5 * w_q * (sm_B.dag() * sm_B - sm_B * sm_B.dag())
H_m = w_m * a_m.dag() * a_m
H_int = g * (sm_A.dag() * a_m + sm_A * a_m.dag()) + \
        g * (sm_B.dag() * a_m + sm_B * a_m.dag())

H = H_q + H_m + H_int

# Collapse Operators (Lindblad Dissipation)
c_ops = [
    np.sqrt(kappa_m * (1 + n_th)) * a_m,
    np.sqrt(kappa_m * n_th) * a_m.dag(),
    np.sqrt(gamma_q) * sm_A,
    np.sqrt(gamma_q) * sm_B
]

# ==========================================
# 3. Initial State & Simulation Time
# ==========================================
# Initial State: |g_A, e_B, 0_m> (Qubit B excited, Bus empty)
psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0), qt.basis(N_m, 0))

# Time evolution span calculated from effective virtual coupling
t_period = np.pi / g_eff
tlist = np.linspace(0, t_period * 1.2, 500) # In nanoseconds

# ==========================================
# 4. Master Equation Solver & Quantum Metrics
# ==========================================
result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[])

pop_A = []
pop_B = []
pop_m = []
entropy_A = []
concurrence_AB = []

for state in result.states:
    # 1. State Populations
    pop_A.append(qt.expect(sm_A.dag() * sm_A, state))
    pop_B.append(qt.expect(sm_B.dag() * sm_B, state))
    pop_m.append(qt.expect(a_m.dag() * a_m, state))

    # 2. Subsystem Von Neumann Entropy S(rho_A)
    rho_A = qt.ptrace(state, 0)
    entropy_A.append(qt.entropy_vn(rho_A, base=2))

    # 3. Distant Qubit Concurrence C_AB
    rho_AB = qt.ptrace(state, [0, 1])
    concurrence_AB.append(qt.concurrence(rho_AB))

print("==================================================")
print(f" Peak Distant Qubit Concurrence (C_AB): {max(concurrence_AB):.4f}")
print(f" Max Magnon Bus Occupancy <n_m>:        {max(pop_m):.4f} (Virtual Exchange)")
print("==================================================")

# ==========================================
# 5. Multi-Panel Publication Plot
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Panel 1: State Transfer Dynamics (Dispersive Regime)
ax1.plot(tlist * 1e3, pop_B, 'r-', lw=2, label=r'Qubit B Population $|e_B\rangle$')
ax1.plot(tlist * 1e3, pop_m, 'g--', lw=1.8, label=r'Virtual Magnon Bus Occupancy $\langle n_m \rangle \approx 0$')
ax1.plot(tlist * 1e3, pop_A, 'b-', lw=2, label=r'Qubit A Population $|e_A\rangle$')
ax1.set_ylabel('Population', fontsize=11)
ax1.set_title('Dispersive Virtual-Magnon Quantum State Transfer', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='center right')

# Panel 2: Subsystem Von Neumann Entropy
ax2.plot(tlist * 1e3, entropy_A, color='purple', lw=2, label=r'Qubit A Entropy $S(\rho_A)$')
ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Maximal Entanglement (1.0 Bit)')
ax2.set_ylabel('Entropy (Bits)', fontsize=11)
ax2.set_title('Qubit-Bus Subsystem Entropy Generation', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# Panel 3: Distant Qubit Concurrence (Reaching C_AB = 1.0)
ax3.plot(tlist * 1e3, concurrence_AB, color='darkorange', lw=2.5, label=r'Distant Concurrence $C_{AB}$')
ax3.axhline(1.0, color='red', linestyle='--', alpha=0.7, label=r'Maximally Entangled Bell State ($C_{AB}=1.0$)')
ax3.set_xlabel('Time (picoseconds)', fontsize=11)
ax3.set_ylabel('Concurrence', fontsize=11)
ax3.set_title('Maximally Entangled State Swapping via Virtual Magnons', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.05, 1.05)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')

plt.tight_layout()
plt.show()