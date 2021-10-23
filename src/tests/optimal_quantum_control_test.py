from unittest.mock import Mock

import pytest
from ..application.optimal_quantum_control import OptimalQuantumControl
from ..config import test_config as config


class TestOptimalQuantumControl:

    def instance_test(self):
        """Tests the instance
        """
        control_params = None
        hamiltonian = None
        time_derivative = None
        target_gate = None
        ex_situ = True

        oqc = OptimalQuantumControl(control_params, hamiltonian, time_derivative, target_gate, ex_situ)

        assert oqc._control_params is control_params
        assert oqc._hamiltonian is hamiltonian
        assert oqc._time_derivative is time_derivative
        assert oqc._target_gate is target_gate
        assert oqc._ex_situ is ex_situ

    def unitary_grape_test(self):
        """Tests the happy path for the unitary GRAPE
        """
        pass
