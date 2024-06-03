#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random parameterized circuit that uses qusim:
e.g. 
    psi = parameterized_qc(psi, nqubit, nlevel=1, thetasByLevels=None)


Created on Mon May 13 11:04:08 2024

@author: adaskin
"""


import numpy as np
from qusim import ry, apply_gate_to_state, prob_of_a_qubit,prob_of_qubits


def init_thetasByLevels(nlevel, nqubit):
    """
    generates random VQC
    return
        U: unitary matrix for VQC
    parameters
        n: #qubits
        level: number of levels in vqc circuit
    """

    rng = np.random.default_rng()
    nqubit2 = int(np.ceil(nqubit / 2))
    thetasByLevels = {}
    itheta = 0  # angles for level
    for lvl in range(nlevel):
        #print("level:", lvl, nlevel)
        #print("Singles=====================")
        thetasByLevels[itheta] = rng.normal(size=(nqubit)) #vector at level itheta
        itheta += 1
        
        #print("Angles for Odd or Evenlevel=====================")
        thetasByLevels[itheta] = rng.normal(size=(nqubit2))
        itheta += 1
        

    return thetasByLevels  # returns thetas in U(theta)
    

####################################################
def apply_rys(psi, nqubit, thetas):
    '''
    applies n ry(thetas[i]) to psi

    Parameters
    ----------
    psi : 1d array
        input state.
    nqubit : int
        number of qubits.
    thetas : 1d array
        angles for n ry gates.
    Returns
    -------
    a quantum state, 1d array
    '''
    
    #layer-0 ry \otimes ry ..\otimes ry
    psi0 =  psi.copy()
    for i in range(nqubit):
        Gate = ry(thetas[i])
        psi0 = apply_gate_to_state(psi0, Gate, i)

        
def parameterized_qc(psi, nqubit, nlevel=1, thetasByLevels=None):
    """
    parameterized quantum circuit with given number of layers
    each layer has 
        1-single gate layer, 
        1-controlled gate layer(controlled gates starts from 0)
        1-controlled gate layer(controlled gates starts from 1)
        that means fully entagling circuit.
    parameters
        n: #qubits
        level: number of levels in vqc circuit
    """
    if thetasByLevels == None:
       thetasByLevels = init_thetasByLevels(nlevel, nqubit)
        
    
    #rng = np.random.default_rng()
    #nqubit2 = int(np.ceil(nqubit/ 2))

    itheta = 0  # level for angles
    for lvl in range(nlevel):
        print("level:", lvl, nlevel)
        thetas = thetasByLevels[itheta] #vector at level itheta
        itheta += 1
        
        
        #single gates on 0, 1, 2,..
        #layer-0 ry \otimes ry ..\otimes ry
        psi0 =  psi.copy()
        for i in range(nqubit):
            Gate = ry(thetas[i])
            psi0 = apply_gate_to_state(psi0, Gate, i)
        
        
        ##CONTROLLED GATES###
        #start from even
        #controlled gates  on (0--1), (2--3), (4--6)...
        psi1 =  psi0.copy()
        for i in range(0, nqubit-1, 2):
            Gate = ry(thetas[i])
            target = i
            control = i+1
            psi1 = apply_gate_to_state(psi1, Gate, target, [control])
        
        #start from odd
        #controlled gates  on (1--2), (3--4), (4--6)...
        psi2 =  psi1.copy()
        for i in range(1, nqubit-1, 2):
            Gate = ry(thetas[i])
            target = i
            control = i+1
            psi2 = apply_gate_to_state(psi2, Gate, target, [control])
        
        
    return psi2  # returns U and theta in U(theta)


   

if __name__ == "__main__":
    nqubit = 4  # number of qubits
    level = 16
    N = 2**nqubit
    rng = np.random.default_rng()
    psi = rng.normal(size=(N,1))
    psi = psi/np.linalg.norm(psi)
    psi = parameterized_qc(psi, nqubit, nlevel=1, thetasByLevels=None)
    qprobs = prob_of_a_qubit(psi,3)
