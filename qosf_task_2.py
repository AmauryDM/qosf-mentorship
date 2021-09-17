import pennylane as qml
import numpy as np
import torch

# input Bloch vector
x = torch.tensor([[0., 0., -1.],
	          [0., 0., 1.],
		  [0., 0., 1.],
		  [0., 0., 1.]])

# target Bloch vector
y = torch.tensor([[0., 0., 1.],
		  [0., 0., 1.],
		  [0., 0., -1.],
		  [0., 0., -1.]])

# array of Pauli matrices
Paulis = torch.zeros([3, 2, 2], dtype=torch.complex128, requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of qubits in the circuit
nr_qubits = 4

# number of layers in the circuit
nr_layers = 1

# randomly initialize parameters from a normal distribution
params = torch.zeros((nr_qubits, 3), requires_grad=True)

# a layer of the circuit ansatz
def layer(params):	
	for i in range(nr_qubits):
		qml.RX(params[i, 0], wires=i)
		qml.RY(params[i, 1], wires=i)
		qml.RZ(params[i, 2], wires=i)

dev = qml.device("default.qubit", wires=nr_qubits)

@qml.qnode(dev, interface="torch")

def circuit(x, params, A):

	# state preparation
	layer(x)

	# hadamard gate at the begining
	for i in range(nr_qubits):
		qml.Hadamard(wires=i)

	# repeatedly apply each layer in the circuit
	layer(params)

	return [qml.expval(qml.Hermitian(A, wires=0)), 
			qml.expval(qml.Hermitian(A, wires=1)),
			qml.expval(qml.Hermitian(A, wires=2)),
			qml.expval(qml.Hermitian(A, wires=3))]

# cost function
def cost_fn(params, index):
	cost = 0
	for k in range(3):
		cost += torch.abs(circuit(x, params, Paulis[k])[index] - y[index][k])
	return cost

# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 50

# optimization with storage of the best parameters
qubit = 2
best_cost = cost_fn(params, qubit)
best_params = torch.zeros((nr_qubits, 3), requires_grad=True)

print("Cost after 0 steps for qubit {} is {:.4f}".format(qubit + 1, best_cost))
# optimization begins
for n in range(steps):
	for i in range(nr_qubits):
		opt.zero_grad()
		loss = cost_fn(params, i)
		loss.backward()
		opt.step()

		# keeps track of best parameters
		if loss < best_cost:
			best_cost = loss
			best_params = params

		# Keep track of progress every 10 steps
		if (n % 10 == 9 or n == steps - 1) and i == qubit:
			print("Cost after {} steps for qubit {} is {:.4f}".format(n + 1, i + 1, loss))

# calculate the Bloch vector of the output state
prediction = torch.zeros((nr_qubits, 3))
for r in range(nr_qubits):
	for c in range(3):
		prediction[r][c] = circuit(x, best_params, Paulis[c])[r]

# print results
print("Target Bloch vector = \n", y)
print("Prediction Bloch vector = \n", prediction)
