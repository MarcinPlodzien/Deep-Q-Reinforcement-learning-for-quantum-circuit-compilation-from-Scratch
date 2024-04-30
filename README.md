# Deep-Q-Reinforcement-learning-for-quantum-circuit-compilation-from-Scratch

Simple and straightforward educational implementation of a Deep-Q-learning and Double-Deep-Q-learning Agent for quantum circuit compilation task.

Code contains:
1. Implementation of the Environment representing quantum state of L-qubit system with set of actions given by predefined quantum gates.
   State of the environment corresponds to quantum state of L-qubits.
   
2. Implementation of the Agent acting on the environment via application of a given quantum gate. Agent applies a chosen gate on the current state
   of the environment and obtain a reward. The reward is "+1" if fidelity between target state and the state of the environment increases, "-1" when decreases,  and "0" when doesn't change. The reward can also be defined as a fidelity between current state and the target state, or change in the fidelity after taken action.

3. Q-table is implemented as a simple feed-forward neural network.

The code runs set of episodes during which the Agent learns. History of each episode is collected in the Pandas dataframe. From dataframe one can extract the optimal set of gates implementing unitary matrix transforming the initial state into the target state with the high fidelity.

In principle, the set of avaiable gates can be arbitrary, thus you can consider connectivity restrictions.
Code can be simply extended to gates with finite fidelity, allowing designing circuits for a target hardware platform.

To learn about theoretical foundations of Reinforcement Learning for Quantum Technologies I strongly suggest our Book
"Modern Applications of Machine Learning in Quantum Sciences": https://arxiv.org/abs/2204.04198

Examples:
1. Preparing 4-qubit |GHZ> state starting from |0000> state with Deep-Q-Learning agent with Hadamard and CZ gates
![image](https://github.com/MarcinPlodzien/Deep-Q-Reinforcement-learning-for-quantum-circuit-compilation-from-Scratch/assets/95550675/616b7685-5033-4f5d-88ac-392d4c4b9e88)




2. Preparing 3-qubit |GHZ> state starting from |000> state with Double-Deep-Q-Learning agent with Hadamard and CNOT gates
![image](https://github.com/MarcinPlodzien/Deep-Q-Reinforcement-learning-for-quantum-circuit-compilation-from-Scratch/assets/95550675/4f628db2-2c78-4e2d-ab21-9a7d45e9dce6)




If you use this code for work that you publish, please cite this repository.



