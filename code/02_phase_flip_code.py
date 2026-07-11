"""
3-Qubit Phase-Flip Code using Qiskit
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile

qc_show = QuantumCircuit(3)
q = QuantumRegister(3, 'q')
anc = QuantumRegister(2, 'anc')
c_anc = ClassicalRegister(2, 'c_anc')
c_data = ClassicalRegister(1, 'c_data')
qc = QuantumCircuit(q, anc, c_anc, c_data)

# ---Original State---

qc_show.h(q[0])
qc_show.cx(q[0], q[1])
qc_show.cx(q[0], q[2])
print("Quantum State:")
display(Statevector(qc_show).draw('latex'))

#---Encoding---

qc.h(q[0])
qc.cx(q[0], q[1])
qc.cx(q[0], q[2])
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.barrier()

#---Error Implementation---

error_qubit = int(input("Enter error qubit (0,1,2): "))
qc.z(error_qubit)
qc.barrier()

#---Converting Phase Error into Bit-flip Error---

qc.h(q[0])
qc.h(q[1])
qc.h(q[2])

# ---Measure Z1Z2---

qc.cx(q[0], anc[0])
qc.cx(q[1], anc[0])

# ---Measure Z2Z3---

qc.cx(q[1], anc[1])
qc.cx(q[2], anc[1])
qc.barrier()
qc.measure(anc, c_anc)

#---Error Correction---

#---Error Correction---

with qc.if_test((c_anc, 2)):
    qc.x(q[2])

with qc.if_test((c_anc, 3)):
    qc.x(q[1])

with qc.if_test((c_anc, 1)):
    qc.x(q[0])

#---Decoding---

qc.cx(q[0], q[2])
qc.cx(q[0], q[1])
qc.barrier()

#---Final Measurement---

qc.measure(q[0], c_data[0])

#-------------------------
simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 5000
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()
print("Raw Syndrome Counts:",counts)
full_output = list(counts.keys())[0]
syndrome = full_output.split(' ')[1]
print("Syndrome =", syndrome)
if syndrome == '00':
    print("No error detected")

elif syndrome == '10':
    print("Error detected on qubit q2")

elif syndrome == '11':
    print("Error detected on qubit q1")

elif syndrome == '01':
    print("Error detected on qubit q0")

qc.draw('mpl',style='iqp-dark')

#---Fidelity Calculation---

count_0 = counts.get('0 11', 0) + counts.get('0 01', 0) + counts.get('0 10', 0) + counts.get('0 00', 0)
count_1 = counts.get('1 11', 0) + counts.get('1 01', 0) + counts.get('1 10', 0) + counts.get('1 00', 0)
p0_exp = count_0 / shots
p1_exp = count_1 / shots
p0_ideal = 0.5
p1_ideal = 0.5
fidelity = ((p0_ideal * p0_exp)**0.5 + (p1_ideal * p1_exp)**0.5 )**2
print("\nExperimental Probabilities:")
print(f"P(0) = {p0_exp:.4f}")
print(f"P(1) = {p1_exp:.4f}")
print("State Fidelity:")
print(f"{fidelity:.6f}")

#---Fidelity Representation---

import matplotlib.pyplot as plt
import numpy as np
mean_exp = p0_exp
mean_ideal = 0.5
std_common = 0.007
x = np.linspace(0.45, 0.55, 500)
gaussian_exp = (1 / (std_common * np.sqrt(2*np.pi))) * np.exp(-((x - mean_exp)**2) /(2 * std_common**2))
gaussian_ideal = (1 / (std_common * np.sqrt(2*np.pi))) * np.exp(-((x - mean_ideal)**2) /(2 * std_common**2))
plt.style.use('dark_background')
plt.figure(figsize=(9,5))
plt.plot(x, gaussian_exp, linewidth=3, color='cyan', label='Experimental Quantum State')
plt.fill_between(x, gaussian_exp, alpha=0.25, color='cyan')
plt.plot(x, gaussian_ideal, linestyle='dashed', linewidth=3, color='red', label='Ideal Quantum State')
plt.axvline(mean_ideal, linestyle='dotted', linewidth=2, color='white', label=f'Ideal Mean={mean_ideal}')
plt.axvline(mean_exp, linestyle='dashdot', linewidth=2, color='yellow', label=f'Experimental Mean={mean_exp:.4f}')
plt.text(0.452, max(gaussian_exp)*0.92, f'Fidelity = {fidelity:.6f}', fontsize=14, color='white')
plt.xlabel('Logical State Measurement Probability', fontsize=14)
plt.ylabel('Quantum State Distribution', fontsize=14)
plt.title('Gaussian Deviation Between Ideal and Noisy Quantum State', fontsize=18)
plt.legend(fontsize=12)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()
