# qusim_torch.py
import torch
import numpy as np

def ry(angle):
    """rotation-y gate - torch compatible"""
    R = torch.zeros(2, 2, dtype=torch.float32)
    half_angle = angle / 2
    R[0, 0] = torch.cos(half_angle)
    R[0, 1] = torch.sin(half_angle)
    R[1, 0] = -torch.sin(half_angle)
    R[1, 1] = torch.cos(half_angle)
    return R

def apply_gate_to_state(psi, Gate, target, control_qubits=[]):
    '''
    applies 2x2 Gate (it may be controlled) to vector psi, 
    qubit orders |0, 1, 2..n> - torch compatible
    '''
    cqsorted = sorted(control_qubits)
    outstate = psi.clone()
    N = len(outstate)
    nqubit = int(np.log2(N))
    
    for j in range(N):#we can also compute a step k from c and t bits
        jbits = bin(j)[2:].zfill(nqubit)
        
        if jbits[target] == '1':  # we already processed when '0'
            continue
        
        skip = False
        for c in cqsorted:
            if jbits[c] == '0':  # control bit is 0
                skip = True
                break

        if not skip:  # all control bits are 1 or no control qubits
            indx1 = (1 << (nqubit - target - 1)) | j  # when target bit 1
            indx0 = j

            psi0_val = Gate[0, 0] * psi[indx0] + Gate[0, 1] * psi[indx1]
            psi1_val = Gate[1, 0] * psi[indx0] + Gate[1, 1] * psi[indx1]
            
            outstate[indx0] = psi0_val
            outstate[indx1] = psi1_val
            
    return outstate

def prob_of_qubits(psi, qubits):
    """Compute probabilities for given qubits - torch compatible"""
    N = len(psi)
    n = int(np.log2(N))
    lenq = len(qubits)
    f = torch.zeros(2**lenq, dtype=torch.float32)

    for j in range(N):
        jbits = bin(j)[2:].zfill(n)
        ind = 0
        for q_idx, q in enumerate(qubits):
            if jbits[q] == "1":
                ind += 2**(lenq - q_idx - 1)
        f[ind] += torch.abs(psi[j])**2
        
    return f

def prob_of_a_qubit(psi, qubit):
    """Compute probability for a single qubit - torch compatible"""
    N = len(psi)
    n = int(np.log2(N))
    f = torch.zeros(2, dtype=torch.float32)
    qshift = n - qubit - 1
    
    for j in range(N):
        qbitval = (j >> qshift) & 1
        f[qbitval] += torch.abs(psi[j])**2
        
    return f
