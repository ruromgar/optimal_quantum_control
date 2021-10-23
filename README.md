# Improving Quantum Gates With Optimal Quantum Control

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
