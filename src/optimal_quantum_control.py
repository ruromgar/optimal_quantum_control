import logging

from .config import config
import numpy as np
from scipy.linalg import expm


class OptimalQuantumControl:
    def __init__(self, control_params, hamiltonian, time_derivative, target_gate, ex_situ=True):
        self._config = config

        self._logger = logging.getLogger(self._config.LOG_CONFIG['name'])
        self._logger.setLevel(self._config.LOG_CONFIG['level'])
        log_handler = self._config.LOG_CONFIG['stream_handler']
        log_handler.setFormatter(logging.Formatter(self._config.LOG_CONFIG['format']))
        self._logger.addHandler(log_handler)

        self._control_params = control_params
        self._hamiltonian = hamiltonian
        self._time_derivative = time_derivative
        self._target_gate = target_gate
        self._ex_situ = ex_situ

    def unitary_grape(self):
        self._logger.info('Calculating unitary GRAPE...')
        # Calculate each unitary
        u_matrix = expm(-1j * self._time_derivative * self._hamiltonian(self._control_params[0]))
        for w in self._control_params[1:]:
            u_matrix = np.matmul(expm(-1j * self._control_params * self._hamiltonian(w)), u_matrix)
        return u_matrix

    def fidelity(self):
        self._logger.info('Calculating fidelity...')
        U=Unitary_grape
        d=V.shape[1]
        f=((abs(np.trace(np.dot(V,U))))**2)/(d*d)
        return f

    def control(self):
        self._logger.info('Optimizing fidelity...')
        pass

    def grape_pulse(self):
        self._logger.info('Calculating GRAPE pulse...')
        pass

