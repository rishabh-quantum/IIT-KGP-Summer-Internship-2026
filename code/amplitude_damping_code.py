import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Kraus
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_bloch_multivector

#-----Define an arbitrary quantum state - |ψ> = cos(θ/2)|0> + exp(iφ)sin(θ/2)|1>-----

theta_deg = float(input("Enter θ (in degrees): "))
phi_deg = float(input("Enter ϕ (in degrees): "))
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)
psi = Statevector([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

#-----Convert to Density Matrix-----

rho_initial = DensityMatrix(psi)
print("\nInitial Density Matrix ρ:")
np.set_printoptions(precision=4, suppress=True)
print(rho_initial.data)

#-----Random Amplitude Damping Parameters-----

gamma = np.random.rand()
print("Amplitude Damping Parameter γ =", f"{gamma:.6f}")

#-----MONTE CARLO SIMULATION-----

N = 10000
rho_avg = np.zeros((2,2),dtype=complex)

for _ in range(N):

    psi_vec = psi.data
    prob_jump = gamma * np.abs(psi_vec[1])**2
    r = np.random.rand()
    
    if r < prob_jump:

        psi_after = Statevector([1,0])

    else:

        K0 = np.array([[1,0], [0,np.sqrt(1-gamma)]], dtype=complex)
        state_after = K0 @ psi_vec
        norm = np.linalg.norm(state_after)

        if norm > 0:
            
            state_after /= norm

        psi_after = Statevector(state_after)

    rho_after = DensityMatrix(psi_after)
    rho_avg += rho_after.data

rho_avg /= N
rho_final = DensityMatrix(rho_avg)

#-----KRAUS AMPLITUDE DAMPING CHANNEL-----

K0 = np.array([[1,0], [0,np.sqrt(1-gamma)]], dtype=complex)
K1 = np.array([[0,np.sqrt(gamma)], [0,0]], dtype=complex)
channel = Kraus([K0,K1])
rho_kraus = rho_initial.evolve(channel)

#-----RESULTS-----

np.set_printoptions(precision=6, suppress=True)
print("\nMonte Carlo Averaged Density Matrix:")
print(rho_avg)
print("\nKraus Channel Density Matrix:")
print(rho_kraus.data)

#-----Purity Checking-----

purity_before = np.real(np.trace(rho_initial.data @ rho_initial.data))
purity_after = np.real(np.trace(rho_final.data @ rho_final.data))
print("Purity Before =", f"{purity_before:.6f}")
print("Purity After  =", f"{purity_after:.6f}")

#-----Von Neumann Entropy-----

eigenvals_before = np.linalg.eigvalsh(rho_initial.data)
eigenvals_after = np.linalg.eigvalsh(rho_final.data)
entropy_before = max(0.0, -np.sum(eigenvals_before*np.log2(eigenvals_before+1e-12)))
entropy_after = max(0.0, -np.sum(eigenvals_after*np.log2(eigenvals_after+1e-12)))
print("\nEntropy Before =", f"{entropy_before:.6f}")
print("Entropy After  =", f"{entropy_after:.6f}")

#-----Fidelity-----

fidelity = state_fidelity(rho_initial,rho_final)
print("\nFidelity =", f"{fidelity:.6f}")

#-----GEOMETRIC ACTION ON BLOCH SPHERE-----

import matplotlib.pyplot as plt
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Depolarizing transformation

x_new = np.sqrt(1-gamma)*x
y_new = np.sqrt(1-gamma)*y
z_new = (1-gamma)*z + gamma
fig = plt.figure(figsize=(9,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, alpha=0.6, edgecolor='k')
ax1.set_title("Original Bloch Sphere")
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_new, y_new, z_new, alpha=0.6, edgecolor='k')
ax2.set_title(f"Amplitude Channel Bloch Sphere (gamma = {gamma:.4f})")

for ax in [ax1, ax2]:

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
