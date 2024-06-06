#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this is the multiprocess version of qusim.py. The usage is the same as qusim.py
TODO: 
    - Needs a little optimization when dividing data into workers
    - some repeated codes can be removed or moved into a function
================================================================
from qusim.py
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

Created on Mon  Jun 5 11:04:08 2024

@author: adaskin
"""

import ctypes
import numpy as np
import threading
from threading import Thread 
import concurrent.futures
import multiprocessing as mp
MAX_THREADS = mp.cpu_count()
#################################################################
def prob_of_a_qubit_serial(psi, qubit):
    """
    computes probabilities in a quantum state for a given qubit.
    Parameters
    ----------
    psi: numpy 1 dimensional column vector
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
#################################################################
#################################################################
def worker_prob_of_a_qubit(psi, start, end, qshift, fshared):

    flocal = np.zeros(2)
    
    for j in range(start, end):
        #jbits = bin(j)[2:].zfill(n)
        #qbitval1 = int(jbits[qubit])
        qbitval = (j >> qshift) & 1
        
        #print(qbitval,qbitval1)
        flocal[qbitval] += np.real(np.abs(psi[j])**2)

    fshared.acquire()
    try:
        for i in range(len(fshared)):
            fshared[i] += flocal[i]
    except:
        print("exception occured while getting lock")
    finally:
        fshared.release()    
    return flocal
############################################
############################################
def prob_of_a_qubit(psi, qubit):
    """uses worker_prob_of_a_qubit() with a shared mem to 
    compute probabilities in a quantum state for a given qubit.
    Parameters
    ----------
    psi: numpy 1 dimensional column vector
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
    fshared = mp.Array(ctypes.c_double, 2, lock=True) 
    
    fzero = np.zeros(2,dtype=ctypes.c_double)

    fshared[:] = fzero[:]
    qshift = n - qubit -1
    
    thread_data_range =  int(N/MAX_THREADS)+1
    if thread_data_range < 1024:
        nthreads = int(N/1024)+1
        thread_data_range = 1024
    else:
        nthreads = MAX_THREADS
    
    
    
    processes = []
    
    #for each thread assign part of the mem
    for ti in range(nthreads):
       
        
        start = thread_data_range*ti
        end = thread_data_range*(ti+1)
        end = N if end > N else end
        p = mp.Process(target=worker_prob_of_a_qubit,
                    args=(psi, start, end, qshift, fshared))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        p.close()


       

    return fshared


#################################################################
#################################################################
def worker_prob_of_qubits(psi, n, qubits, start, end, fshared):
    '''computes prob in range psi[start:end] 
    for qubits
    fshared is used for return value
    '''
    lenq = len(qubits)
    flocal = np.zeros(2**lenq)
    
    for j in range(start, end):
        jbits = bin(j)[2:].zfill(n)
        ind = 0
        for q in range(lenq):
            if jbits[qubits[q]] == "1":
                ind += 2**(lenq - q-1)

        flocal[ind] += np.real(np.abs(psi[j])**2)
        
    fshared.acquire()
    try:
        for i in range(len(fshared)):
            fshared[i] += flocal[i]
    except:
        print("exception occured while getting lock")
    finally:
        fshared.release()    
    return flocal

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
   
    fshared = mp.Array(ctypes.c_double, 2**lenq, lock=True) 
    fzero = np.zeros(2**lenq,dtype=ctypes.c_double)
    fshared[:] = fzero[:]

    thread_data_range =  int(N/MAX_THREADS)+1
    if thread_data_range < 1024:
        nthreads = int(N/1024)+1
        thread_data_range = 1024
    else:
        nthreads = MAX_THREADS
    
    
    
    processes = []

    
    #for each thread assign part of the mem
    for ti in range(nthreads):
 
        
        start = thread_data_range*ti
        end = thread_data_range*(ti+1)
        end = N if end > N else end
        p = mp.Process(target=worker_prob_of_qubits, 
                       args=(psi, n, qubits, start, end, fshared))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        p.close()
    
    return fshared

def prob_of_qubits_serial(psi, qubits):
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


def worker_apply_gate_to_state(psi_shared, 
                               start, end, 
                               Gate, target, cqsorted, 
                               nqubit, qshift, 
                               lock=None ):
    '''computes prob in range psi[start:end] and psi[where target is 1]
    for qubit binary-masked by qshift
    psi_shared is a shared memory
    '''
    # #memcpy to not affect psi0, psi1 computations 
    # changed0states = []#target is 0: these are unique between processes
    # changed1states = []#target is 1: these are unique between processes
        
    
    
    for j in range(start, end):
        
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
            indx1 = (1<<(qshift)) | j #when target bit 1
            indx0 = j
    
            #no lock needed since indx0 indx1 are uniques
            psi0 = Gate[0][0]*psi_shared[indx0] + Gate[0][1]*psi_shared[indx1]
            psi1 = Gate[1][0]*psi_shared[indx0] + Gate[1][1]*psi_shared[indx1]
            psi_shared[indx0] = psi0
            psi_shared[indx1] = psi1
            #changed0states.append([indx0,psi0])
            #changed1states.append([indx1,psi1])


    return #changed0states,changed1states

def apply_gate_to_state(psi_shared, Gate, target, control_qubits=[]):
    '''
    applies 2x2 Gate (it may be controlled) to vector psi, 
    qubit orders |0, 1, 2..n>
    Parameters
    ----------
    psi: one dimensional column vector 
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
    N = len(psi_shared)
    nqubit = int(np.log2(N))

    qshift = (nqubit-target-1)
    nqubit = int(np.log2(N))

    thread_data_range =  int(N/MAX_THREADS)+1
    if thread_data_range < 1024:
        nthreads = int(N/1024)+1
        thread_data_range = 1024
    else:
        nthreads = MAX_THREADS
    
    
    
    processes = []
    lock = mp.Lock()
    

    #for each thread assign part of the mem
    for ti in range(nthreads):
        #TODO divide data only where target=0
        start = thread_data_range*ti
        end = thread_data_range*(ti+1)
        end = N if end > N else end
        p = mp.Process(target= worker_apply_gate_to_state, 
                       args=(psi_shared, 
                            start, end, 
                            Gate, target, cqsorted, 
                            nqubit, qshift,lock))
        processes.append(p)
        p.start()
    


    for p in processes:
        p.join()
        p.close()
   
    return psi_shared

def apply_gate_to_state_serial(psi, Gate, target, control_qubits=[]):
    '''
    applies 2x2 Gate (it may be controlled) to vector psi, 
    qubit orders |0, 1, 2..n>
    Parameters
    ----------
    psi: one dimensional column vector 
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
    
    N = len(psi)
    outstate = np.zeros(N)
    outstate[:] = psi[:]
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
    from timeit import default_timer as timer
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    nqubit = 20  # number of qubits
    COMPARE_WITH_SERIAL = True #to run serial or not
    N = 2**nqubit
    rng = np.random.default_rng()
   
    ##initial array is UNLOCKED: 
    ##because lock is not necessary 
    ##in READINGS such as probability of qubits etc.
    psi = mp.Array(ctypes.c_double, N, lock=False) # shared
    
    #psi = np.frombuffer(buffer,dtype=ctypes.c_double, count=N)
    if COMPARE_WITH_SERIAL == True:
        test_data = rng.normal(size=(N))
        test_data = test_data/np.linalg.norm(test_data)   
        psi[:] = test_data[:]
    else:
        psi[:] = rng.normal(size=(N))
        psi[:] = psi/np.linalg.norm(psi)


    gate = ry(rng.normal())
    ##############################################
    #To test serial vs mp 
    if COMPARE_WITH_SERIAL == True:
        start=timer()
        initial_qprobs = prob_of_a_qubit_serial(psi,3)
        psi1 = apply_gate_to_state_serial(psi, gate, 3, [0, 1,2])
        
        final_qprobs = prob_of_a_qubit_serial(psi1,3)
        
        qp0 = prob_of_qubits_serial(psi,[0,3])
        qp1 = prob_of_qubits_serial(psi1,[0,3])
        end=timer()
       
        # for i in range(len(psi)):
        #      print(psi[i], psi1[i])
        print("Applied gate:\n", gate)
        print("init prob of 4th qubit:", initial_qprobs)
        print("final prob of 4th qubit:", final_qprobs)
    
    
        print("init prob of 1st and 4th qubit:", qp0)
        print("final prob of 1st and 4th qubit:",qp1)
        print("serial-time:", end - start) 
        psi[:] = test_data[:]
        
    #######################################
    start2=timer()
    initial_qprobs = prob_of_a_qubit(psi,3)
    qp0 = prob_of_qubits(psi,[0,3])
    
    psi1 = apply_gate_to_state(psi, gate, 3, [0, 1,2])

    final_qprobs = prob_of_a_qubit(psi1,3)
    qp1 = prob_of_qubits(psi1,[0,3])
    end2=timer()

    # for i in range(len(psi)):
    #     print(psi[i], psi1[i])
    print("Applied gate:\n", gate)
    print("init prob of 4th qubit:", initial_qprobs[:])
    print("final prob of 4th qubit:", final_qprobs[:])
     
    print("init prob of 1st and 4th qubit:", qp0[:])
    print("final prob of 1st and 4th qubit:",qp1[:])
    
    print("mp-time:", end2 - start2) 
    pr.disable()
    #pr.print_stats()