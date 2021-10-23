import numpy as np
import random as rm
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp
from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister, Aer         
from qiskit.providers import Backend, BaseBackend
from qiskit.utils import QuantumInstance    

def Outer2Kron( A, Dims ):
    """
    From (vec(A) outer vec(B)) to (A kron B)
    """
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape(2*Dims), np.array([range(N),range(N,2*N) ]).T.flatten() ).flatten()
    return A.reshape(Dim)

def Kron2Outer( A, Dims ):
    """
    From (A kron B) to (vec(A) outer vec(B))
    """
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape( np.kron(np.array([1,1]),Dims) ), np.array([range(0,2*N,2),range(1,2*N,2)]).flatten() ).flatten()
    return A.reshape(Dim)

def LocalProduct( Psi, Operators , Dims=[] ):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims: 
        Dims = [ Operators[k].shape[-1] for k in range( len(Operators) ) ]
    N = len(Dims)
    for k in range(N):
        Psi  = (( Operators[k]@Psi.reshape(Dims[k],-1) ).T ).flatten()
    return Psi

def InnerProductMatrices( X, B, Vectorized = False ):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)
    
    if isinstance(B, list): 
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False :
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2 :
                    B[j] = np.array([B[j]])
                nops.append( B[j].shape[0] )
                Dims.append( B[j].shape[1] )
                B[j] = B[j].reshape(nops[j],Dims[j]**2)
        elif Vectorized == True :
            for j in range(nsys):
                nops.append( B[j].shape[0] )
                Dims.append( int(np.sqrt(B[j].shape[1])) )                
        
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 
        
        return np.array( TrXB ).reshape(nops)
        
    elif isinstance(B, np.ndarray):     
        
        if B.ndim == 2 and Vectorized == False :
            return np.trace( X @ B )
        
        elif B.ndim == 4 :
            nsys = B.shape[0]
            nops = nsys*[ B[0].shape[0] ]
            Dims = nsys*[ B[0].shape[1] ]
            B = B.reshape(nsys,nops[0],Dims[0]**2)
            
        elif B.ndim == 3 :
            if Vectorized == False :
                nsys = 1
                nops = B.shape[0]       
                Dims = [ B.shape[1] ]
                B = B.reshape(nsys,nops,Dims[0]**2)
            if Vectorized == True :
                nsys = B.shape[0]
                nops = nsys*[ B[0].shape[0] ]
                Dims = nsys*[ int(np.sqrt(B[0].shape[1])) ]
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 

        return np.array( TrXB ).reshape(nops)

def convert_to_base(decimal_number, base, fill=0):
    remainder_stack = []
    DIGITS = '0123456789abcdef'
    while decimal_number > 0:
        remainder = decimal_number % base
        remainder_stack.append(remainder)
        decimal_number = decimal_number // base

    new_digits = []
    while remainder_stack:
        new_digits.append(DIGITS[remainder_stack.pop()])

    return ''.join(new_digits).zfill( fill )

def SimulateMeasurement( rho , Measures , sample_size=0 , output=0 ):
    
    probabilities = np.abs(  InnerProductMatrices( rho, Measures ) ).flatten()
    #probabilities = probabilities/np.sum( probabilities )
    if sample_size > 0:
        probabilities = np.random.multinomial( sample_size, probabilities  )
    if output == 0:
        return probabilities
    else:
        return probabilities/np.sum(probabilities)            

def ExpectedValuePauli( ρ, Labels, shots=0, n=None ):
    """
    Simulate the measurement of a multiqubit Pauli operator. 
    """
    if not isinstance(Labels, list):
        Labels = [Labels]
        
    if n is None: 
        n = len(Labels[0])
        
    if isinstance(shots,int) : 
        shots = len(Labels)*[shots]
    
    POVM = np.array( [
                    [ [ 1., 0., 0, 0. ], [ 0., 0., 0., 1. ] ],
                    [ [ 1./2, 1./2, 1./2, 1./2 ], [ 1./2, -1./2, -1./2, 1./2 ] ], 
                    [ [ 1./2, -1.j/2, 1.j/2, 1./2 ], [ 1./2, 1.j/2., -1j/2, 1./2 ] ], 
                    [ [ 1., 0., 0., 0. ], [ 0., 0., 0., 1. ] ] 
                    ] ).reshape(4,2,2,2)
    EigenValues = [
                np.array([ 1.,  1. ]),
                np.array([ 1., -1. ]),
                np.array([ 1., -1. ]),
                np.array([ 1., -1. ])
                ]
    
    ExpVal = []
    
    for label in Labels:
        if label == ''.zfill(n) :
            ExpVal.append( 1 )
        else:
            probs = SimulateMeasurement( ρ , 
                                        [ POVM[int(label[k])] for k in range(n)  ], 
                                        shots[k], 1 )
            ExpVal.append( LocalProduct( probs , 
                            [ EigenValues[int(label[k])] for k in range(n) ]  )[0] )
            
    return np.array( ExpVal )

def Direct_Fidelity_Estimation( U, V, M, *args  ): 
    """
    Direct Fidelity Estimation (DFE).  
    Estimate the fidelity <ψ|ρ|ψ> measuring M Pauli Operators. 
    
    Inputs: 
        U     : Qiskit circuit, 2^n qubits general process. 
        V     : np.array, 2^n qubits unitary gate. 
        M     : Number of observables for the DFE. 
        *args : Extra inputs of the fun function.
    Output: 
        Fid : Estimated Fidelity. 
    """
    
    if isinstance(ρ, QuantumCircuit):
        fun = Expected_Value_Qiskit
    else:
        fun = None
    
    σ = np.outer( V.flatten(), V.flatten().conj() )
    n = int( np.log2( σ.shape[0] ) )
    paulis = np.array( [ [1,0,0,1], [0,1,1,0],[0,-1j,1j,0],[1,0,0,-1] ] ).reshape(4,2,2) / np.sqrt(2) 
    Pk = n*[ paulis ]
    σk = np.real( InnerProductMatrices( σ, Pk ) ).flatten()
    qk = np.real(σk**2)
    Index = rm.choices( range(4**n), qk, k = M )
    
    if fun is None:
        ρ   = np.outer( U.flatten(), U.flatten().conj() )
        ρk  = np.real( InnerProductMatrices( ρ, Pk ) ).flatten()
        ρki = ρk[Index] 
    else:
        Index_b4 = [ convert_to_base( index, 4, n) for index in Index]
        ρki = fun( ρ, Index_b4, *args )/np.sqrt(2**n) 
        
    σki = σk[Index] 
    Fid = np.real( np.sum( ρki/σki )/M ) 
    return Fid


def get_probabilities( Job ):
    """
    Get the probabilities from a qiskit Job.
    """
    
    results = Job.results
    
    num_qubits = results[0].header.memory_slots
    
    num_circuits = len( results )
    
    counts = np.zeros( [ 2**num_qubits, num_circuits ] )
    
    for j in range( num_circuits ):
        counts_dic = results[j].data.counts
        for b in counts_dic:
            if b[1]=='x':
                bb = int(b,0)
            else:
                bb = int(b,2)
            counts[ bb , j ] = counts_dic[ b ]
        
    prob_exp = counts/np.sum(counts,0)
    
    return prob_exp

def Expected_Value_Qiskit( circ_U , Labels, quantum_instance, shots=2**13 ):

    if isinstance(quantum_instance, Backend) or isinstance(quantum_instance, BaseBackend):
        quantum_instance = QuantumInstance(backend = quantum_instance, shots=shots)
    
    n = circ_U.num_qubits
    M = len(Labels)
    
    if not isinstance(Labels, list):
        Labels = [Labels]
        
    EigenValues = [
                np.array([ 1.,  1. ]),
                np.array([ 1., -1. ]),
                np.array([ 1., -1. ]),
                np.array([ -1., 1. ])
                ] 
    
    Circuits = []
    Labels_out = []
    ExpectedValues = M*[None]
    Indx = []
    
    for m in range(M):
        label = Labels[m]
        if label == ''.zfill(n):
            ExpectedValues[m] = 1  
        else:
            Indx.append(m)
            Labels_out.append(label)
            circuit_0 = QuantumCircuit(n,n)
            circuit_0.compose( circ_ρ, qubits=range(n), inplace=True) 
            circuit_0.barrier()
            
            circuit_1 = circuit_0.copy()
            
            n_mid = int(n/2)
            
            for k in range(n_mid):
                idex = label[::-1][k]
                if idex == '0':
                    circuit_1.x(k)
                elif idex == '1':
                    circuit_0.h(k)
                    circuit_1.x(k)
                    circuit_1.h(k)
                elif idex == '2':
                    circuit_0.u2( 0, -np.pi/2, k )
                    circuit_1.x(k)
                    circuit_1.u2( 0, -np.pi/2, k )
            for k in range(n_mid,n):
                idex = label[::-1][k]
                if idex == '1':
                    circuit_0.h(k)
                    circuit_1.h(k)
                elif idex == '2':
                    circuit_0.u2( 0, -np.pi/2, k )
                    circuit_1.u2( 0, -np.pi/2, k )
            circuit.measure( range(n), range(n) )
            Circuits.append( circuit )
    
    Job = quantum_instance.execute( Circuits )
    
    probs = get_probabilities( Job )
    
    for j in range( len(Labels_out) ):
        ExpectedValues[Indx[j]] = LocalProduct( probs[:,j].T , 
                                  [ EigenValues[int(Labels_out[j][k])] for k in range(n) ]  )[0] 
                                     
    
    return np.array(ExpectedValues)

def Fidelity(state1,state2):
    """
    Calculate the Fidelity between two arbitrary quantum states.
    """
    n1 = state1.ndim
    n2 = state2.ndim
    if n1 == n2:
        if n1 == 1:
            fidelity = np.abs(np.vdot(state1,state2))**2
        else:
            temp     = la.sqrtm(state2)
            temp     = 0.5*(temp+np.conj(temp).T)
            temp     = la.sqrtm(temp@state1@temp)
            fidelity = np.trace(0.5*(temp+np.conj(temp).T))**2
    else:
        if n1==1:
            fidelity = np.vdot(state1,state2@state1)
        else:
            fidelity = np.vdot(state2,state1@state2)
    return np.real(fidelity)