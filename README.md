# Improving Quantum Gates With Optimal Quantum Control

## Abstract
It is necessary to improve the fidelity of quantum gates to archive computational advantage with quantum computers. An alternative to do that is to use optimal quantum control. This is based on optimizing the parameters of a Hamiltonian to maximize the fidelity with a target quantum gate. The evaluation of the fidelity can be carried out numerically (ex-situ) or experimentally (in-situ). In this project, we propose implementing a Qiskit library to perform optimal quantum control.

Our objective is to build a Qiskit library to perform ex-situ and in-situ optimal quantum control. Our control parameters will be the amplitudes of a GRAPE (Gradient Ascent Pulse Engineering) pulse, which will be implemented using Qiskit Pulse. The experimental evaluation of the fidelity will be performed by Direct Fidelity Estimation. We also propose a mixed protocol, where first the ex-situ quantum control is carried out, to then refine the result with in-situ quantum control. We expect to implement some relevant gate with our routines, such as NOT-gate or Hadamard.

## Members
- Luciano Pereira Valenzuela
- Rafael González López
- Miguel Ángel Palomo Marcos
- Alejandro Bravo
- Rubén Romero García

## Entregable
Github repository with the code to perform optimal quantum control. Include some examples for a single-qubit.

This project is available in TestPypi under https://test.pypi.org/project/oqc/

`pip install -i https://test.pypi.org/simple/ oqc==1.0.1`

Basic usage so far (requires Python3 and virtualenv): 

```
cd optimal_quantum_control
virtualenv -p python3 venv
pip install -r requirements.txt
```

And then, from inside a Python shell (or in a script or a notebook):

```
from oqc.optimal_quantum_control import OptimalQuantumControl
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_armonk')

# Some random values, change accordingly
initial_control_params = [1, 2, 3]
time_derivative = 1
target_gate = np.array([[0, 1], [1, 0]])
ex_situ = True

oqc = OptimalQuantumControl(initial_control_params, backend, time_derivative, target_gate, ex_situ)
result = oqc.control()
print(result)
```

Tests

```
cd optimal_quantum_control
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pytest --cov
```
