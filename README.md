# Deep-Q-Reinforcement-learning-for-quantum-state-compilation-from-Scratch

Simple and straightforward educational implementation of a deep Q-learning agent for quantum circuit compilation task.

Code contains:
1. Implementation of the Environment representing quantum state of L-qubit system with set of actions given by predefined quantum gates.
   State of the environment corresponds to quantum state of L-qubits.
   
2. Implementation of the Agent acting on the environment via application of a given quantum gate. Agent applies a chosen gate on the current state
   of the environment and obtain a reward. The reward is "+1" if fidelity between target state and the state of the environment increases, "-1" when decreases,   
   and "0" when doesn't change.

3. Q-table is implemented as a simple feed-forward neural network.

The code runs set of episodes during which the Agent learns. History of each epsiode is collected in the Pandas dataframe. From dataframe one can extract the optimal set of gates implementing unitary transforming the initial state into the target state with the highes fidelity.

Example output for the preparing two-qubit |GHZ> state starting with initial |11> state.

![image](https://github.com/MarcinPlodzien/Deep-Q-Reinforcement-learning-for-quantum-state-compilation-from-Scratch/assets/95550675/8e63e8d8-9726-4654-8ace-54960efb83f5)

