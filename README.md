# Deep-Q-Reinforcement-learning-for-quantum-state-compilation-from-Scratch

Simple and straightforward educational implementation of a Deep-Q-learning and Double-Deep-Q-learning agent for quantum circuit compilation task.

Code contains:
1. Implementation of the Environment representing quantum state of L-qubit system with set of actions given by predefined quantum gates.
   State of the environment corresponds to quantum state of L-qubits.
   
2. Implementation of the Agent acting on the environment via application of a given quantum gate. Agent applies a chosen gate on the current state
   of the environment and obtain a reward. The reward is "+1" if fidelity between target state and the state of the environment increases, "-1" when decreases,   
   and "0" when doesn't change.

3. Q-table is implemented as a simple feed-forward neural network.

The code runs set of episodes during which the Agent learns. History of each epsiode is collected in the Pandas dataframe. From dataframe one can extract the optimal set of gates implementing unitary transforming the initial state into the target state with the highes fidelity.

To learn about theoretical foundations of Reinforcement Learning for Quantum Technologies I strongly suggest our Book
"Modern Applications of Machine Learning in Quantum Sciences": https://arxiv.org/abs/2204.04198

Examples:
1. Preparing 2-qubit |GHZ> state starting from |00> state with Deep-Q-Learning agent
![image](https://github.com/MarcinPlodzien/Deep-Q-Reinforcement-learning-for-quantum-state-compilation-from-Scratch/assets/95550675/a31a1cb7-8ef5-4ff6-b53c-0973435b0d2e)




2. Preparing 3-qubit |GHZ> state starting from |000> state with Double-Deep-Q-Learning agent
   ![image](https://github.com/MarcinPlodzien/Deep-Q-Reinforcement-learning-for-quantum-state-compilation-from-Scratch/assets/95550675/85fdf54e-d365-4b4f-be24-b7946be8376e)


