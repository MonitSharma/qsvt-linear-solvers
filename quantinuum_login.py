import qnexus
import numpy as np
from primitives.qsp_qsvt import approximate_inverse, find_phases
from hardware.quantinuum_runner import build_qsvt_circuit, NexusRunner

A = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=complex)
b = np.array([1.0, 0.0], dtype=complex)
poly = approximate_inverse(kappa=2.0, epsilon=0.3)
phis = find_phases(poly.cheb_coeffs, poly.parity)
circ = build_qsvt_circuit(A, phis, b, measure=True)

runner = NexusRunner(project_name="qsvt-helios-demo")
runner.upload(circ, name="qsvt-2x2-inverse")

# Cheapest stage only — confirm the fix before spending on noisy/hardware
lite = runner.run_emulator_lite(n_shots=200)
print("Helios-1E-lite counts:", lite.counts)