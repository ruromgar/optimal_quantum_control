# Improving Quantum Gates With Optimal Quantum Control

This project is available in TestPypi under https://test.pypi.org/project/oqc/

`pip install -i https://test.pypi.org/simple/ oqc`

Basic usage so far (requires Python3 and virtualenv): 

```
cd optimal_quantum_control
virtualenv -p python3 venv
python3 main.py
```

Tests

```
cd optimal_quantum_control
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pytest --cov
```
