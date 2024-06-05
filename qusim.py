#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this files includes a simple quantum simulator 

the qubit index order:|0, 1, ..n>

the main simulator: apply_gate_to_state(psi, Gate, target, control_qubits=[])
e.g.
            Gate = ry(theta)# a 2x2 matrix
            
            psi1 = apply_gate_to_state(psi, Gate, 3, [1,2])

measurement of qubits: prob_of_qubits(psi, qubits)
e.g.
    qporbs = prob_of_a_qubit(psi,4)
    qporbs = prob_of_qubits(psi,[4,5,6])

random parameterized circuit as vectors:
e.g. 
    psi = parameterized_qc(psi, nqubit, nlevel=1, thetasByLevels=None)

Created on Mon May 13 11:04:08 2024

@author: adaskin
"""


import numpy as np



############################################
#################################################################
def prob_of_qubits(psi, qubits):
    """
    computes probabilities for the states of a given qubit list.
    Parameters
    ----------
    psi: numpy 1 dimensional column vector
        representing a quantum state
    qubits:  list
        the set of qubits such as [0 3 5]
        - the order of the qubits |0, 1,...,n>

    Returns
    -------
    numpy ndarray
        a vector that represents probabilities.
    """

    N = len(psi)

    n = int(np.log2(N))
    lenq = len(qubits)
    f = np.zeros(2**lenq)

    for j in range(N):
        jbits = bin(j)[2:].zfill(n)
        ind = 0
        for q in range(lenq):
            if jbits[qubits[q]] == "1":
                ind += 2**(lenq - q-1)

        f[ind] += np.real(np.abs(psi[j])**2)
    return f


############################################
#################################################################
def prob_of_a_qubit(psi, qubit):
    """
    computes probabilities in a quantum state for a given qubit.
    Parameters
    ----------
    psi: numpy 1 dimensional row vector
        representing a quantum state
    qubit:  int
        an integer number
        - the order of the qubits |0,1,..n-1>

    Returns
    -------
    numpy ndarray
        a vector that represents probabilities.
    """
    N = len(psi)

    n = int(np.log2(N))

    f = np.zeros(2)
    qshift = n - qubit -1
    for j in range(N):
        #jbits = bin(j)[2:].zfill(n)
        #qbitval1 = int(jbits[qubit])
        qbitval = (j >> qshift) & 1
        
        #print(qbitval,qbitval1)
        f[qbitval] += np.real(np.abs(psi[j])**2)
    return f


####################################################


def ry(angle):
    """rotation-y gate"""
    R = np.zeros([2, 2])
    angle = angle / 2
    R[0, 0] = np.cos(angle)
    R[0, 1] = np.sin(angle)
    R[1, 0] = -np.sin(angle)
    R[1, 1] = np.cos(angle)
    return R



def apply_gate_to_state(psi, Gate, target, control_qubits=[]):
    '''
    applies 2x2 Gate (it may be controlled) to vector psi, 
    qubit orders |0, 1, 2..n>
    Parameters
    ----------
    psi: one dimensional row vector 
        a quantum state
    Gate: 2x2 numpy ndarray
        A quantum gate
    target: int
        target qubit
    control_qubits: []
        list of control qubits
        qubit orders |0, 1, 2..n>
    Returns
    -------
    a new quantum state

    '''
    cqsorted = sorted(control_qubits)
    outstate = psi.copy()
    N = len(outstate)
    nqubit = int(np.log2(N))
    for j in range(N):
        jbits = bin(j)[2:].zfill(nqubit)
        
        if jbits[target] == '1': #we already processed when '0'
            continue
        
        skip = False
        for c in cqsorted:
            if jbits[c] == '0': #control bit is 0
                skip = True
                break
 
        if skip == False:#control bit is 0
            #when, all control is 1  or no control qubits
            indx1 = (1<<(nqubit-target-1)) | j #when target bit 1
            indx0 = j
    
            #print(indx0, indx1, bin(indx0)[2:].zfill(nqubit), bin(indx1)[2:].zfill(nqubit))
            psi0 = Gate[0][0]*psi[indx0] + Gate[0][1]*psi[indx1]
            psi1 = Gate[1][0]*psi[indx0] + Gate[1][1]*psi[indx1]
            outstate[indx0] = psi0
            outstate[indx1] = psi1
    return outstate



   
if __name__ == "__main__":
    nqubit = 4  # number of qubits
    level = 16
    N = 2**nqubit
    rng = np.random.default_rng()
    psi = rng.normal(size=(N))
    psi = psi/np.linalg.norm(psi)
    gate = ry(rng.normal())
    psi1 = apply_gate_to_state(psi, gate, 3, [0, 1,2])
    initial_qprobs = prob_of_a_qubit(psi,3)
    final_qprobs = prob_of_a_qubit(psi1,3)

    for i in range(len(psi)):
        print(psi[i], psi1[i])
    print("Applied gate:\n", gate)
    print("init prob of 4th qubit:", initial_qprobs)
    print("final prob of 4th qubit:", final_qprobs)

    qp0 = prob_of_qubits(psi,[0,3])
    qp1 = prob_of_qubits(psi1,[0,3])
    print("init prob of 1st and 4th qubit:", qp0)
    print("final prob of 1st and 4th qubit:",qp1)