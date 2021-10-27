import logging

import numpy as np
from noisyopt import minimizeSPSA
from qiskit import QuantumCircuit, pulse
from qiskit.test.mock import FakeValencia
from qiskit.circuit import Gate
from qiskit import transpile, schedule as build_schedule
from scipy.linalg import expm
from scipy.optimize import minimize

from .config import config


class OptimalQuantumControl:
    def __init__(
            self,
            initial_control_params,
            backend,
            time_derivative,
            target_gate,
            ex_situ: bool = True
    ) -> None:
        self._config = config

        self._logger = logging.getLogger(self._config.LOG_CONFIG['name'])
        self._logger.setLevel(self._config.LOG_CONFIG['level'])
        log_handler = self._config.LOG_CONFIG['stream_handler']
        log_handler.setFormatter(logging.Formatter(self._config.LOG_CONFIG['format']))
        self._logger.addHandler(log_handler)

        self._initial_control_params = initial_control_params
        self._backend = backend
        self._time_derivative = time_derivative
        self._target_gate = target_gate
        self._ex_situ = ex_situ

    def unitary_grape(self, control_params):
        """Calculates the unitary matrix according to GRAPE.

        Parameters
        -------
        control_params
            Array of control parameters

        Returns
        -------
        Unitary matrix
        """

        # self._logger.info('Calculating unitary GRAPE...')
        u_matrix = expm(-1j * self._time_derivative * self.calculate_hamiltonian(control_params[0]))
        for w in control_params[1:]:
            u_matrix = np.matmul(expm(-1j * self._time_derivative * self.calculate_hamiltonian(w)), u_matrix)
        return u_matrix

    def fidelity(self, control_params):
        """Calculates the infidelity between the target gate and the unitary matrix.

        Parameters
        -------
        control_params
            Array of control parameters

        Returns
        -------
        Infidelity measure
        """

        self._logger.info('Calculating fidelity...')
        d2 = np.power(self._target_gate.shape[1], 2)
        fid = 1 - ((abs(np.trace(np.matmul(self._target_gate.T.conj(), self.unitary_grape(control_params))))) ** 2) / d2
        print(fid)
        return fid
    
    def fidelity_experimental(self, control_params):
        """Calculates the experimental infidelity between the target gate and the unitary matrix by direct fidelity estimation.

        Parameters
        -------
        control_params
            Array of control parameters

        Returns
        -------
        Inverse fidelity measure
        """

        self._logger.info('Calculating fidelity on IBM-Q...')
        schedule = self.grape_pulse(control_params)
        fid = 1 - Direct_Fidelity_Estimation( schedule, self._target_gate, 20, self._backend )
        
        print(fid)
        return fid    
    
    def control(self):
        """Maximizes the fidelity between the target gate and the unitary matrix,
        finding the optimal array of control parameters.

        Returns
        -------
        Optimal array of control params
        """

        self._logger.info('Optimizing fidelity...')
        bounds = [(0, 1) for _ in range(len(self._initial_control_params))]
        if self._ex_situ:
            opt = minimize(self.fidelity, x0=self._initial_control_params, bounds=bounds)
        else:
            opt = minimizeSPSA(self.fidelity_experimental, x0=self._initial_control_params, bounds=bounds, paired = False, niter=100)
        return opt.x
    
    def control_mixed(self):
        """Maximizes the fidelity between the target gate and the unitary matrix,
        finding the optimal array of control parameters mixizing the ex-situ and in-situ quantum control.

        Returns
        -------
        Optimal array of control params
        """

        self._logger.info('Optimizing fidelity...')
        bounds = [(0, 1) for _ in range(len(self._initial_control_params))]
        opt = minimize(self.fidelity, x0=self._initial_control_params, bounds=bounds)
        opt = minimizeSPSA(self.fidelity_experimental, x0=opt.x, bounds=bounds, paired = False, niter=100)
        return opt.x        
    

    def grape_pulse(self, control_params):
        """Calibrate a single gate in a single qbit circuit.

        Parameters
        -------
        control_params
            Array of control parameters

        Returns
        -------
        Circuit with calibrated gate
        """
        self._logger.info('Calculating GRAPE pulse...')
#         cir = QuantumCircuit(1, 1)
#         cir.h(0)
#         cir.measure(0, 0)

# #         backend = FakeValencia()
#         backend = self._backend
#         with pulse.build(backend, name='hadamard') as h_q0:
#             for w in control_params:
#                 pulse.play(pulse.Constant(self._time_derivative, w), pulse.DriveChannel(0))

#         cir.add_calibration('h', [0], h_q0)
        cir = QuantumCircuit(1,1)
        custom_gate = Gate('Grape_Gate', 1, control_params )
        cir.append(custom_gate, [0])
        backend = self._backend
        with pulse.build(backend, name='Grape_Pulse') as h_q0:
            for w in control_params:
                pulse.play(pulse.Constant(self._time_derivative, w), pulse.DriveChannel(0))

        cir.add_calibration('Grape_Gate', [0], h_q0)

        return h_q0

    def calculate_hamiltonian(self, dt):
        """Calculates the hamiltonian matrix

        Parameters
        -------
        dt
            Hamiltonian constant Dt

        Returns
        -------
        Hamiltonian matrix
        """

        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        identity = np.array([[1, 0], [0, 1]])
        amplitude = 1 #self._backend.configuration().hamiltonian['vars']['omegad0'] / 10e9
        frequency = 1 #self._backend.configuration().hamiltonian['vars']['wq0']  / 10e9

        return ((1 / 2) * frequency * (identity - pauli_z)) + (amplitude * dt * pauli_x)


############### Direct fidelity estimation ##############
    
import numpy as np
import random as rm
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
    """"
    Transform a decimal number to any basis.
    """
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
        U     : Qiskit pulse, 2^n qubits general process. 
        V     : np.array, 2^n qubits unitary gate. 
        M     : Number of observables for the DFE. 
        *args : Extra inputs of the fun function.
    Output: 
        Fid : Estimated Fidelity. 
    """
    
#     if isinstance(U, QuantumCircuit):
#         fun = Expected_Value_Pulse
#     else:
#         fun = None

    fun = Expected_Value_Pulse
    
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
        ρki = fun( U, Index_b4, *args )
        
    σki = σk[Index] 
    Fid = np.real( np.sum( ρki/σki )/M ) 
    return Fid


def get_probabilities( Job, M ):
    """
    Get the probabilities from a qiskit Job.
    """
    
    results = Job.result()
    
    num_qubits = 1
    
    num_circuits = M
    
    counts = np.zeros( [ 2**num_qubits, num_circuits ] )
    
    for j in range( num_circuits ):
        counts_dic = results.get_counts(j)
        for b in counts_dic:
            bb = int(b,2)
            counts[ bb , j ] = counts_dic[ b ]
        
    prob_exp = counts/np.sum(counts,0)
    
    return prob_exp

def Expected_Value_Qiskit( circ_U , Labels, quantum_instance, shots=2**13 ):
    """
    Calculate tr( Y [ sigma_i x sigma_j ] ) experimentally, with Y the Choi operator associated with
    the single qubit circuit circ_U,
    In:
        circ_U: qiskit circuit.
        Labels: str. It specifies the Pauli operators.
        quantum_instance: quantum instance
        shots: number of shots.
    Out:
        np.array(ExpectedValues) : np.array. Expected values tr( Y [ sigma_i x sigma_j ] ). 
    """
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
                np.array([ 1., -1. ])
                ] 

    Circuits = []
    Labels_out = []
    ExpectedValues = M*[None]
    Indx = []

    for m in range(M):
        label = Labels[m]
        if label == ''.zfill(2*n):
            ExpectedValues[m] = 1  
        else:
            Indx.append(m)
            Labels_out.append(label)
            circuit_0 = QuantumCircuit(n,n)
            circuit_1 = QuantumCircuit(n,n)

            if label[1] == '0' or label[1] == '3' :
                circuit_0.barrier()
                circuit_1.x(0)
                circuit_1.barrier()

            elif label[1] == '1':
                circuit_0.h(0)
                circuit_0.barrier()
                circuit_1.x(0)
                circuit_1.h(0)
                circuit_1.barrier()

            elif label[1] == '2' : 
                circuit_0.x(0)
                circuit_0.u2(np.pi/2,np.pi,0) 
                circuit_0.barrier()
                circuit_1.u2(np.pi/2,np.pi,0) 
                circuit_1.barrier()

            circuit_0.compose( circ_U, qubits=range(n), inplace=True) 
            circuit_0.barrier()
            circuit_1.compose( circ_U, qubits=range(n), inplace=True) 
            circuit_1.barrier()

            if label[0] == '1':
                circuit_0.h(0)
                circuit_1.h(0)
            elif label[0] == '2' :
                circuit_0.u2(0,np.pi/2,0) 
                circuit_1.u2(0,np.pi/2,0) 

            circuit_0.measure(range(n),range(n))
            circuit_1.measure(range(n),range(n))

            Circuits.append( circuit_0 )       
            Circuits.append( circuit_1 )       

    
    Job = quantum_instance.execute( Circuits )
    
    probs = get_probabilities( Job )
    
    for k in range(len(Labels_out)):
        if Labels_out[k][0] == '0':
            S0 = probs[1,2*k] + probs[0,2*k]
            S1 = probs[1,2*k+1] + probs[0,2*k+1]
        else:
            S0 = - probs[1,2*k] + probs[0,2*k]
            S1 = - probs[1,2*k+1] + probs[0,2*k+1]

#         print(S0, S1)
        if Labels_out[k][1] == '0':
            ExpectedValues[ Indx[k]] = S1/2. + S0/2.
        else:
            ExpectedValues[ Indx[k]] = -S1/2. + S0/2.
    
    ExpectedValues = np.array(ExpectedValues)
    
    return ExpectedValues

def circuit2schedule(circuit, backend):
    transpiled_circuit = transpile(circuit, backend)  
    schedule = build_schedule(transpiled_circuit, backend)
    return schedule

def Schedule2Block(Schedule):
    blockschedule = pulse.ScheduleBlock()
    if len(Schedule.instructions) > 0 :
        for a in Schedule.instructions[0]:
            if a != 0:
                blockschedule += a
    return blockschedule

def Expected_Value_Pulse( schedule_U , Labels, backend, shots=2**13 ):
    """
    Calculate tr( Y [ sigma_i x sigma_j ] ) experimentally, with Y the Choi operator associated with
    the single qubit circuit circ_U,
    In:
        circ_U: qiskit schedule pulse.
        Labels: str. It specifies the Pauli operators.
        quantum_instance: quantum instance
        shots: number of shots.
    Out:
        np.array(ExpectedValues) : np.array. Expected values tr( Y [ sigma_i x sigma_j ] ). 
    """
    
    n = 1
    M = len(Labels)

    if not isinstance(Labels, list):
        Labels = [Labels]

    EigenValues = [
                np.array([ 1.,  1. ]),
                np.array([ 1., -1. ]),
                np.array([ 1., -1. ]),
                np.array([ 1., -1. ])
                ] 

    Circuits = []
    Labels_out = []
    ExpectedValues = M*[None]
    Indx = []

    for m in range(M):
        label = Labels[m]
        if label == ''.zfill(2*n):
            ExpectedValues[m] = 1  
        else:
            Indx.append(m)
            Labels_out.append(label)
            circuit_0 = QuantumCircuit(n,n)
            circuit_1 = QuantumCircuit(n,n)

            if label[1] == '0' or label[1] == '3' :
                circuit_1.x(0)

            elif label[1] == '1':
                circuit_0.h(0)
                circuit_1.x(0)
                circuit_1.h(0)
                
            elif label[1] == '2' : 
                circuit_0.x(0)
                circuit_0.u2(np.pi/2,np.pi,0) 
                circuit_1.u2(np.pi/2,np.pi,0) 
            
            schedule_0 = Schedule2Block( circuit2schedule(circuit_0, backend) )
            schedule_1 = Schedule2Block( circuit2schedule(circuit_1, backend) )
            schedule_0.append( schedule_U )
            schedule_1.append( schedule_U )

            circuit_0 = QuantumCircuit(n,n)
            circuit_1 = QuantumCircuit(n,n)
            
            if label[0] == '1':
                circuit_0.h(0)
                circuit_1.h(0)
            elif label[0] == '2' :
                circuit_0.u2(0,np.pi/2,0) 
                circuit_1.u2(0,np.pi/2,0) 

            circuit_0.measure(range(n),range(n))
            circuit_1.measure(range(n),range(n))
            
            schedule_0.append( Schedule2Block( circuit2schedule(circuit_0, backend)) )
            schedule_1.append( Schedule2Block( circuit2schedule(circuit_1, backend)) )
            
            Circuits.append( schedule_0 )       
            Circuits.append( schedule_1 )     
                
    Job = backend.run( Circuits, shots=shots, meas_level=2 )
    
    probs = get_probabilities( Job, len(Circuits) )
        
    for k in range(len(Labels_out)):
        if Labels_out[k][0] == '0':
            S0 = probs[1,2*k] + probs[0,2*k]
            S1 = probs[1,2*k+1] + probs[0,2*k+1]
        else:
            S0 = - probs[1,2*k] + probs[0,2*k]
            S1 = - probs[1,2*k+1] + probs[0,2*k+1]

        if Labels_out[k][1] == '0':
            ExpectedValues[ Indx[k]] = S1/2. + S0/2.
        else:
            ExpectedValues[ Indx[k]] = -S1/2. + S0/2.
    
    ExpectedValues = np.array(ExpectedValues)
    
    return ExpectedValues
