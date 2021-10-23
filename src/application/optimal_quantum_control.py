import logging
import numpy as np

from ..config import config
from scipy.linalg import expm
from scipy.optimize import minimize

# from qiskit.algorithms.optimizers.optimizer import SPSA


class OptimalQuantumControl:
    def __init__(
            self,
            initial_control_params,
            hamiltonian,
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
        self._hamiltonian = hamiltonian
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

        self._logger.info('Calculating unitary GRAPE...')
        u_matrix = expm(-1j * self._time_derivative * self._hamiltonian(control_params[0]))
        for w in self._initial_control_params[1:]:
            u_matrix = np.matmul(expm(-1j * control_params * self._hamiltonian(w)), u_matrix)
        return u_matrix

    def fidelity(self, control_params):
        """Calculates the fidelity between the target gate and the unitary matrix.

        Parameters
        -------
        control_params
            Array of control parameters

        Returns
        -------
        Fidelity measure
        """

        self._logger.info('Calculating fidelity...')
        d = self._target_gate.shape[1]
        return ((abs(np.trace(np.dot(self._target_gate, self.unitary_grape(control_params))))) ** 2) / (d * d)

    def control(self):
        """Maximizes the fidelity between the target gate and the unitary matrix,
        finding the optimal array of control parameters.

        Returns
        -------
        Optimal array of control params
        """

        self._logger.info('Optimizing fidelity...')
        return minimize(
            lambda w: 1 - self.fidelity(w),
            x0=self._initial_control_params,
            bounds=[(0, 1) for _ in range(len(self._initial_control_params))]
        )
        # TODO: use SPSA as optimizer
        # self._optimum_control_params = 1 - SPSA.optimize(
        #    num_vars=len(self._initial_control_params), objective_function=self.fidelity)

    def grape_pulse(self):
        """TBD

        Returns
        -------
        TBD
        """

        self._logger.info('Calculating GRAPE pulse...')
        pass
