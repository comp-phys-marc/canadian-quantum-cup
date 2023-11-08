import json
import pennylane as qml
import pennylane.numpy as np

wires_m = [0, 1, 2]  # qubits needed to encode m
wires_n = [3, 4, 5]  # qubits needed to encode n
wires_aux = [6, 7, 8, 9, 10]  # auxiliary qubits you can use


# Put your code here #

# Create all the helper functions you need here
def fourier_basis(wire_list,d):
    qml.QFT(wires=wire_list)
    for i in range(len(wire_list)):
        const = d*np.pi/(2**i)
        qml.PhaseShift(const ,wires=[i])
    qml.adjoint(qml.QFT(wires=wire_list))
    
def inverse_fourier_basis(wire_list,d):
    qml.QFT(wires=wire_list)
    for i in range(len(wire_list)):
        const = -d*np.pi/(2**i)
        qml.PhaseShift(const ,wires=[i])
    qml.adjoint(qml.QFT(wires=wire_list))
    

def oracle_distance(d):
    """
    Args:
        d (int): the distance with which we will check that the oracle is working properly.

    This function does not return anything, it is a quantum function that applies
    necessary gates that implement the requested oracle.

    """
    qml.PauliX(9)
    qml.Hadamard(9)
    # state_1+d==state_2?
    fourier_basis([0,1,2],d)
    
    # check if state_1[i]==state_2[i]
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    # all 0s
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    # all 0s
    
    qml.MultiControlledX([6,7,8], 9)
    
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    # all 0s
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    # all 0s
    
    inverse_fourier_basis([0,1,2],d)
    qml.Hadamard(9)
    qml.PauliX(9)
    
    #======================================#
    
    qml.PauliX(9)
    qml.Hadamard(9)
    # state_2+d=state_1
    fourier_basis([3,4,5],d)
    
    # check if state_1[i]==state_2[i]
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    # all 0s
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    # all 0s
    
    qml.MultiControlledX([6,7,8], 9)
    
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    # all 0s
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    qml.Toffoli([0,3,6])
    qml.Toffoli([1,4,7])
    qml.Toffoli([2,5,8])
    for i in range(len([0,3,1,4,2,5])):
        qml.PauliX(i)
    # all 0s
    
    inverse_fourier_basis([3,4,5],d)

    qml.Hadamard(9)
    qml.PauliX(9)
    # Put your code here


# These functions are responsible for testing the solution.
wires_m = [0, 1, 2]
wires_n = [3, 4, 5]
wires_aux = [6, 7, 8, 9, 10]

dev = qml.device("default.qubit", wires=11)


@qml.qnode(dev)
def circuit(m, n, d):
    qml.BasisEmbedding(m, wires=wires_m)
    qml.BasisEmbedding(n, wires=wires_n)
    oracle_distance(d)
    return qml.state()


def run(test_case_input: str) -> str:
    outputs = []
    d = int(json.loads(test_case_input))
    for n in range(8):
        for m in range(8):
            outputs.append(sum(circuit(n, m, d)).real)
            print(n,m,d,circuit(n, m, d))
    outputs.append(d)
    output_list = [elem for elem in outputs[:-1]] + [outputs[-1]]
    
    return str(output_list)


def check(solution_output: str, expected_output: str) -> None:
    i = 0
    solution_output = json.loads(solution_output)
    d = solution_output[-1]
    assert expected_output == "No output", "Something went wrong"
    for n in range(8):
        for m in range(8):
            solution = 1
            if abs(n - m) == d:
                solution = -1
            assert np.isclose(solution_output[i], solution)
            i += 1

    circuit(np.random.randint(7), np.random.randint(7), np.random.randint(7))
    tape = circuit.qtape

    names = [op.name for op in tape.operations]

    assert names.count("QubitUnitary") == 0, "Can't use custom-built gates!"


# These are the public test cases
test_cases = [
    ('0', 'No output'),
    ('1', 'No output'),
    ('2', 'No output'),
    ('3', 'No output'),
    ('4', 'No output'),
    ('5', 'No output'),
    ('6', 'No output'),
    ('7', 'No output')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
