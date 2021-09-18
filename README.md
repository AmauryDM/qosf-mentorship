# QOSF Mentorship Task 2

In this task, the aim is to train a variational circuit to get from a given input state to one of the four proposed. 
To do so, I chose the following corresponding states for each output state.

```markdown
circuit(|1000>) = |0011>
circuit(|0100>) = |0101>
circuit(|0010>) = |1010>
circuit(|0001>) = |1100>
```

From this, I used the Bloch vector representation for each 4 qubit considered by the situation.
Thus, it appears as follows: a line refers to a given qubit and the columns repspectively to the X, Y and Z coordinates in the Bloch sphere. 

```markdown
[[0., 0., -1.],
 [0., 0., 1.],
 [0., 0., 1.],
 [0., 0., 1.]]
-> |1000>
```

With this representation for the qubit, I used the RX, RY and RZ operations to construct the variational circuit.
The variational circuit updates the parameters of the rotational operations.

After training for the given 4 qubit state and the corresponding target state, the result gives the comparison between the desired state and the prediction from the circuit.

```markdown
circuit([[0., 0., -1.],
         [0., 0., 1.],
         [0., 0., 1.],
         [0., 0., 1.]])

---------------- RESULTS ----------------

Target Bloch vector =
[[ 0.,  0.,  1.],
 [ 0.,  0.,  1.],
 [ 0.,  0., -1.],
 [ 0.,  0., -1.]]

Prediction Bloch vector =
[[-1.8904e-02,  8.6253e-05,  9.9982e-01],
 [ 3.3159e-02,  4.7529e-04,  9.9945e-01],
 [ 4.8799e-02,  5.3623e-04, -9.9881e-01],
 [ 6.5838e-02, -3.5558e-04, -9.9783e-01]]
```

This result is given for a cost of 2% for each measurement of the different qubits given by the Pauli matrices on each axis.

Once trained for this specific dataset, it is possible to use the following state to test the model on unseen data.

```markdown
[[0., 0., -1.],
 [0., 0., -.],
 [0., 0., -.],
 [0., 0., -.]]
-> |1111>
```

The result is the following state which is close to one of the output states from the dataset.

```markdown
Test Bloch vector =
[[ 0.0346,  0.0341, -0.9988],
 [ 0.0389,  0.0379, -0.9985],
 [ 0.0437, -0.0421,  0.9982],
 [ 0.0490, -0.0469,  0.9977]]
 ```

An explanation is that the model chooses an output state based on how much the input state is similar to one of the input states from the dataset. It means the model has totally overfitted the data to perfectly give the desired output based on the corresponding input. Thus when given another different state, the model tries to map it to an output state based on how close the input might be from a state in the dataset. In this case, the model mapped it as follows.

```markdown
circuit(|1111>) ~ circuit(|0001>) = |1100>
```
