"""
Bit-flip Channel code using Qiskit
"""

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Kraus
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_bloch_multivector

#-----Define an arbitrary quantum state - |ψ> = cos(θ/2)|0> + exp(iφ)sin(θ/2)|1>-----

theta_deg = float(input("Enter θ (in degrees): "))
phi_deg = float(input("Enter φ (in degrees): "))
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)
psi = Statevector([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

#-----Convert to Density Matrix-----

rho_initial = DensityMatrix(psi)
print("Initial Density Matrix ρ:")
np.set_printoptions(precision=4, suppress=True)
print(rho_initial.data)

#-----Random Bit Flip Probability-----

p = np.random.rand()
print("Bit Flip Probability p =", f"{p:.6f}")

#-----MONTE CARLO SIMULATION-----

I = np.array([[1,0], [0,1]], dtype=complex)
X = np.array([[0,1], [1,0]], dtype=complex)
operations = [I,X]
probabilities = [1-p,p]
N = 10000
rho_avg = np.zeros((2,2),dtype=complex)

for _ in range(N):

    chosen = np.random.choice([0,1],p=probabilities)
    psi_after = psi.evolve(Operator(operations[chosen]))
    rho_after = DensityMatrix(psi_after)
    rho_avg += rho_after.data

rho_avg /= N
rho_final = DensityMatrix(rho_avg)

#-----KRAUS BIT FLIP CHANNEL-----

K0 = np.sqrt(1-p)*I
K1 = np.sqrt(p)*X
channel = Kraus([K0,K1])
rho_kraus = rho_initial.evolve(channel)

#-----RESULTS-----

np.set_printoptions(precision=6,suppress=True)
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
entropy_before = max(0.0, -np.sum(eigenvals_before * np.log2(eigenvals_before + 1e-12)))
entropy_after = max(0.0, -np.sum(eigenvals_after * np.log2(eigenvals_after + 1e-12)))
print("Entropy Before =", f"{entropy_before:.6f}")
print("Entropy After  =", f"{entropy_after:.6f}")

#-----Fidelity-----

fidelity = state_fidelity(rho_initial, rho_final)
print("Fidelity =", f"{fidelity:.6f}")

#-----GEOMETRIC ACTION ON BLOCH SPHERE-----

import matplotlib.pyplot as plt
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Depolarizing transformation

factor = 1 - p
x_new = x
y_new = (1-2*p)*y
z_new = (1-2*p)*z
fig = plt.figure(figsize=(9,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, alpha=0.6, edgecolor='k')
ax1.set_title("Original Bloch Sphere")
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_new, y_new, z_new, alpha=0.6, edgecolor='k')
ax2.set_title(f"Bit-flip Channel (p = {p:.4f})")

for ax in [ax1, ax2]:

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
