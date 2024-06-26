# a simple quantum circuit simulator and its multiprocess version
## ``qusim.py`` 
includes a simple quantum circuit simulator that applies a given gate to a quantum state.

- The qubit index order: ``|0, 1, ..n>``

The main simulator function is: 
``apply_gate_to_state(psi, Gate, target, control_qubits=[])``
e.g.
```
Gate = ry(theta)# a 2x2 matrix

psi1 = apply_gate_to_state(psi, Gate, 3, [1,2])
```
## Measurement of qubits: 
```
prob_of_qubits(psi, qubits)
prob_of_a_qubit(psi,qubit)
```
e.g.
```
qporbs = prob_of_a_qubit(psi,4)
qporbs = prob_of_qubits(psi,[4,5,6])
```

## ``example_parametized_circuit.py``

 shows application of qusim and generates a random nlevel parameterized circuit.
e.g. 
```
psi = parameterized_qc(psi, nqubit, nlevel=1, thetasByLevels=None)
```

## ``qusimmp.py`` Multiprocess/multithread implementation
multiprocess implementation of qusim on shared memory. The functions are the same as ``qusim.py`` and it can be used as ``qusim.py``.
Note that this is not completely optimized. It may include a few bugs. 
