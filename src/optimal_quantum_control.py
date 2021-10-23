import logging

from .config import config


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
        pass

    def fidelity(self):
        self._logger.info('Calculating fidelity...')
        pass

    def control(self):
        self._logger.info('Optimizing fidelity...')
        pass

    def grape_pulse(self):
        self._logger.info('Calculating GRAPE pulse...')
        pass

