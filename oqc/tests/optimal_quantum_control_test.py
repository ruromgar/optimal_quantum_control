from unittest.mock import Mock

from optimal_quantum_control.oqc.optimal_quantum_control import OptimalQuantumControl

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
        hamiltonian = None
        time_derivative = None
        target_gate = np.array([[1, 0], [0, 1]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, hamiltonian, time_derivative, target_gate, ex_situ)
        oqc.unitary_grape = Mock(return_value=np.array([[1, 0], [0, 1]]))

        control_params = None
        result = oqc.fidelity(control_params)

        assert result == 1

    def fidelity_orthogonal_matrices_test(self):
        """Tests the happy path for the fidelity: if matrices are
        orthogonal, fidelity should equal to 0
        """
        initial_control_params = None
        hamiltonian = None
        time_derivative = None
        target_gate = np.array([[0, 1], [1, 0]])
        ex_situ = True

        oqc = OptimalQuantumControl(initial_control_params, hamiltonian, time_derivative, target_gate, ex_situ)
        oqc.unitary_grape = Mock(return_value=np.array([[1, 0], [0, 1]]))

        control_params = None
        result = oqc.fidelity(control_params)

        assert result == 0

    def control(self):
        """Tests the happy path for the optimizer
        """
        pass

    def grape_pulse(self):
        """Tests the happy path for the GRAPE pulse
        """
        pass
