from unittest.mock import Mock, patch

from optimal_quantum_control.oqc.optimal_quantum_control import OptimalQuantumControl
from optimal_quantum_control.oqc import optimal_quantum_control as oqc_module

import numpy as np


class TestOptimalQuantumControl:

    def instance_test(self):
        """Tests the instance
        """
        initial_control_params = None
        backend = None
        time_derivative = None
        target_gate = None
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)

        assert oqc._initial_control_params is initial_control_params
        assert oqc._backend is backend
        assert oqc._time_derivative is time_derivative
        assert oqc._target_gate is target_gate
        assert oqc._ex_situ is ex_situ

    def unitary_grape_test(self):
        """Tests the happy path for the unitary GRAPE
        """
        pass

    def fidelity_identical_matrices_test(self):
        """Tests the happy path for the fidelity: if matrices are
        identical, fidelity should equal to 1
        """
        initial_control_params = None
        backend = None
        time_derivative = None
        target_gate = np.array([[1, 0], [0, 1]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)
        oqc.unitary_grape = Mock(return_value=np.array([[1, 0], [0, 1]]))

        control_params = None
        result = oqc.fidelity(control_params)

        assert result == 1

    def fidelity_orthogonal_matrices_test(self):
        """Tests the happy path for the fidelity: if matrices are
        orthogonal, fidelity should equal to 0
        """
        initial_control_params = None
        backend = None
        time_derivative = None
        target_gate = np.array([[0, 1], [1, 0]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)
        oqc.unitary_grape = Mock(return_value=np.array([[1, 0], [0, 1]]))

        control_params = None
        result = oqc.fidelity(control_params)

        assert result == 0

    def control_test(self):
        """Tests the happy path for the optimizer
        """
        initial_control_params = [0.1, 0.2, 0.3]
        backend = None
        time_derivative = None
        target_gate = np.array([[1, 0], [0, 1]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)
        oqc.fidelity = Mock(return_value=0.5)

        result = oqc.control()

        assert np.array_equal(result, initial_control_params) is True

    def grape_pulse(self):
        """Tests the happy path for the GRAPE pulse
        """
        pass

    def calculate_hamiltonian_test(self):
        """Tests the hamiltonian calculation
        """
        initial_control_params = None
        backend = Mock()
        time_derivative = None
        target_gate = np.array([[1, 0], [0, 1]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)
        oqc._backend.properties = Mock()
        oqc._backend.properties.return_value.frequency = Mock(return_value=1)

        dt = 3
        result = oqc.calculate_hamiltonian(dt)
        expected_result = np.array([[0, 3], [3, 1]], dtype=float)

        assert np.array_equal(result, expected_result) is True
