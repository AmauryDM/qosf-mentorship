import pennylane as qml
import numpy as np
import torch

# input Bloch vectors
x1 = torch.tensor([[0., 0., -1.],
				   [0., 0., 1.],
				   [0., 0., 1.],
				   [0., 0., 1.]])

x2 = torch.tensor([[0., 0., 1.],
				   [0., 0., -1.],
				   [0., 0., 1.],
				   [0., 0., 1.]])

x3 = torch.tensor([[0., 0., 1.],
				   [0., 0., 1.],
				   [0., 0., -1.],
				   [0., 0., 1.]])

x4 = torch.tensor([[0., 0., 1.],
				   [0., 0., 1.],
				   [0., 0., 1.],
				   [0., 0., -1.]])

X = [x1, x2, x3, x4]

# target output Bloch vectors
y1 = torch.tensor([[0., 0., 1.],
				   [0., 0., 1.],
				   [0., 0., -1.],
				   [0., 0., -1.]])

y2 = torch.tensor([[0., 0., 1.],
				   [0., 0.,- 1.],
				   [0., 0., 1.],
				   [0., 0., -1.]])

y3 = torch.tensor([[0., 0., -1.],
				   [0., 0., 1.],
				   [0., 0., -1.],
				   [0., 0., 1.]])

y4 = torch.tensor([[0., 0., -1.],
				   [0., 0., -1.],
				   [0., 0., 1.],
				   [0., 0., 1.]])

Y = [y1, y2, y3, y4]

# array of Pauli matrices for measurements
Paulis = torch.zeros([3, 2, 2], dtype=torch.complex128, requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of qubits in the circuit
nr_qubits = 4

# number of layers in the circuit
nr_layers = 4

# initialize tensor of parameters to zeros
params = torch.zeros((nr_layers, nr_qubits, 3), requires_grad=True)

# preparation of the input state
def state_preparation(x):
	for i in range(nr_qubits):
		qml.RX(x[i, 0], wires=i)
		qml.RY(x[i, 1], wires=i)
		qml.RZ(x[i, 2], wires=i)

# a layer of the circuit ansatz
def layer(params, index):
	for i in range(nr_qubits):
		qml.RX(params[index, i, 0], wires=i)
		qml.RY(params[index, i, 1], wires=i)
		qml.RZ(params[index, i, 2], wires=i)

dev = qml.device("default.qubit", wires=nr_qubits)

@qml.qnode(dev, interface="torch")

def circuit(x, params, A):

	# state preparation
	state_preparation(x)

	# hadamard gate for superposition
	for i in range(nr_qubits):
		qml.Hadamard(wires=i)

	# repeatedly apply each layer in the circuit
	for j in range(nr_layers):
		layer(params, j)

	return [qml.expval(qml.Hermitian(A, wires=0)), 
			qml.expval(qml.Hermitian(A, wires=1)),
			qml.expval(qml.Hermitian(A, wires=2)),
			qml.expval(qml.Hermitian(A, wires=3))]

# cost function
def cost_fn(params, index, x, y):
	cost = 0
	for k in range(3):
		cost += torch.abs(circuit(x, params, Paulis[k])[index] - y[index][k])
	return cost

# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of epochs in the optimization routine
epochs = 4

# number of steps for each epoch
steps = 40

# prepare tensors for different predictions
predictions = []
for i in range(len(X)):
	predictions.append(torch.zeros((nr_qubits, 3)))

# setup part
for epoch in range(epochs):
	print("\nEpoch {}".format(epoch))

	# go through all the dataset for training
	for batch_index in range(len(X)):
		print("\nInput and target states {}".format(batch_index+1))
		x_batch = X[batch_index]
		y_batch = Y[batch_index]

		# random qubit to print the cost
		qubit = np.random.randint(0, nr_qubits)

		# storage of the best variables
		best_cost = cost_fn(params, qubit, x_batch, y_batch)
		best_params = torch.zeros((nr_layers, nr_qubits, 3), requires_grad=True)
		print("Cost after 0 steps for qubit {} is {:.4f}".format(qubit+1, best_cost))

		# optimization part
		for n in range(steps):
			# train for each qubit
			for i in range(nr_qubits):
				opt.zero_grad()
				loss = cost_fn(params, i, x_batch, y_batch)
				loss.backward()
				opt.step()

				# keep track of best parameters
				if loss < best_cost:
					best_cost = loss
					best_params = params

				# keep track of progress every 10 steps
				if (n % 10 == 9 or n == steps - 1) and i == qubit:
					print("Cost after {} steps for qubit {} is {:.4f}".format(n+1, i+1, loss))

		# calculate the Bloch vector of the output state
		for r in range(nr_qubits):
			for c in range(3):
				predictions[batch_index][r][c] = circuit(X[batch_index], best_params, Paulis[c])[r]

# print results
for i in range(len(X)):
	print("Target Bloch vector =\n", Y[i])
	print("Prediction Bloch vector =\n", predictions[i])
