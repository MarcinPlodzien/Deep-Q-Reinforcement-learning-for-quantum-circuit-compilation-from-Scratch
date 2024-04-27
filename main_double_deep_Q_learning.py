import numpy as np
import torch as pt
from torch import matrix_exp as expm
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import copy


def get_Identity(k):  # returns k-tensor product of the identity operator, ie. Id^k
    Id = id_local
    for i in range(0, k-1):
        Id = pt.kron(Id, id_local)
    return Id
       
def get_chain_operator(A, L, i):
    Op = A
    if(i == 1):
        Op = pt.kron(A,get_Identity(L-1))
        return Op
    if(i == L):
        Op = pt.kron(get_Identity(L-1),A)
        return Op
    if(i>0 and i<L):
        Op = pt.kron(get_Identity(i-1), pt.kron(Op, get_Identity(L-i)))
        return Op

def get_chain_operators(L):
    
    Id = get_chain_operator(id_local, L, 1)
    X = {}
    Y = {}
    Z = {}
    Hadamard = {}
    for qubit_i in range(1, L+1):                                    
        X[qubit_i] = get_chain_operator(sigma_x, L, qubit_i)       
        Y[qubit_i] = get_chain_operator(sigma_y, L, qubit_i)        
        Z[qubit_i] = get_chain_operator(sigma_z, L, qubit_i)       
        Hadamard[qubit_i] = get_chain_operator(hadamard, L, qubit_i)       
    return Id, X, Y, Z, Hadamard

id_local = pt.tensor([[1.,0],[0,1.]], dtype=pt.complex64)
sigma_x = pt.tensor([[0,1.],[1.,0]], dtype=pt.complex64)
sigma_y = 1j*pt.tensor([[0,-1.],[1.,0]], dtype=pt.complex64)
sigma_z = pt.tensor([[1.,0],[0,-1.]], dtype=pt.complex64)
hadamard = 1.0/pt.sqrt(pt.tensor(2))*pt.tensor([[1,1],[1,-1]], dtype=pt.complex64)+1j*0    

def get_range(i,j):
    return np.arange(i,j+1)

class QuantumCircuitEnvironment:
    def __init__(self, L, psi_initial, psi_target,  max_steps, fidelity_threshold):
        
        self.L = L                      # Number of qubits
        self.D = 2**L                   # Hilbert space size
        self.state_dimension = 2**(L+1) # doubled size for (Re[psi], Im[psi])

        self.max_steps = max_steps
        self.fidelity_threshold = fidelity_threshold
        self.current_step = 0
        
        self.psi_initial = psi_initial      # initial state of qubits
        self.psi_target = psi_target        # target quantum state
        self.psi_current = self.psi_initial
        self.norm = pt.sum(pt.abs(self.psi_current)**2)        
        self.phase_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #discretization for phase of gates, in units of pi/2
        self.action_space_basic_gates = [(gate_symbol, qubit_idx, -1) for gate_symbol in ["Id", "H" ] for qubit_idx in get_range(1, L)]  
        self.action_space_global_rotations = [(gate_symbol, -1, phase) for gate_symbol in [ "Rx_global", "Ry_global", "Rz_global"]  for phase in self.phase_grid] 
        self.action_space_local_rotations  = [(gate_symbol, i, phase) for gate_symbol in [ "Rx_local", "Ry_local", "Rz_local"] for i in get_range(1, L) for phase in self.phase_grid] 
        self.action_space_entangling_parametrized_gates     = [(gate_symbol, i, phase) for gate_symbol in [  "Rxx", "Ryy", "Rzz"] for i in get_range(1, L) for phase in self.phase_grid] 
        self.action_space_entangling_non_parametrized_gates = [(gate_symbol, qubit_idx, -1) for gate_symbol in ["CNOT"] for qubit_idx in get_range(1, L)]  
        self.action_space = []
        self.action_space += self.action_space_basic_gates
        # self.action_space += self.action_space_global_rotations
        # self.action_space += self.action_space_local_rotations
        # self.action_space += self.action_space_entangling_parametrized_gates
        self.action_space += self.action_space_entangling_non_parametrized_gates
 
        self.Id, self.X, self.Y, self.Z, self.Hadamard = get_chain_operators(L)
        self.CNOT = {}
        self.X_total = sum([self.X[i] for i in get_range(1,L)])
        self.Y_total = sum([self.Y[i] for i in get_range(1,L)])
        self.Z_total = sum([self.Z[i] for i in get_range(1,L)])
        
        for i in get_range(1, L):
            if(i<L):
                j = i+1                
            if(i==L):
                j = 1
            self.CNOT[i] = expm(pt.pi/4*(self.Id - self.X[i])@(self.Id - self.Z[j])*1j)
                

    def get_gate(self, gate_symbol, i, theta):
        
        if(gate_symbol == "X"):
            return self.X[i]
        
        elif(gate_symbol == "Y"):
            return self.Y[i]
        
        elif(gate_symbol == "Z"):
            return self.Z[i]
        
        elif(gate_symbol == "H"):
            return self.Hadamard[i]
 
        elif(gate_symbol == "Rx_local"):
            return expm(-1j*theta*pt.pi*self.X[i])
        
        elif(gate_symbol == "Ry_local"):
            return expm(-1j*theta*pt.pi*self.Y[i])
        
        elif(gate_symbol == "Rz_local"):
            return expm(-1j*theta*pt.pi*self.Z[i])

        elif(gate_symbol == "Rx_global"):
            return expm(-1j*theta*pt.pi*self.X_total)
        
        elif(gate_symbol == "Ry_global"):
            return expm(-1j*theta*pt.pi*self.Y_total)
        
        elif(gate_symbol == "Rz_global"):
            return expm(-1j*theta*pt.pi*self.Z_total)
        
        elif(gate_symbol == "Rxx"):
            if(i<L):
                j = i+1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])
            if(i==L):
                j = 1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])

        elif(gate_symbol == "Ryy"):
            if(i<L):
                j = i+1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])
            if(i==L):
                j = 1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])

        elif(gate_symbol == "Rzz"):
            if(i<L):
                j = i+1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])
            if(i==L):
                j = 1
                return expm(-1j*theta*pt.pi*self.X[i]@self.X[j])            

        elif(gate_symbol == "CNOT"):
            return self.CNOT[i]
    
        elif(gate_symbol == "Id"):
            return self.Id
        
        return self.Id  

    def reset(self):
        self.current_step = 0
        return self.psi_initial   
    
    def get_fidelity(self, psi_target, psi_current):
        psi_target_tensor = pt.tensor(self.psi_target, dtype=pt.complex64)
        psi_current_tensor = pt.tensor(psi_current, dtype=pt.complex64)
        fidelity = pt.abs(pt.vdot(psi_target_tensor, psi_current_tensor))**2
        return fidelity.item()    

    def step(self, action):
        gate_symbol, qubit_idx, theta = action
        quantum_gate = self.get_gate(gate_symbol, qubit_idx, theta)
        psi_next = quantum_gate @ self.psi_current
        new_fidelity = pt.abs(pt.vdot(self.psi_target, psi_next))**2
        
        # Compare fidelity with the previous state
        previous_fidelity = pt.abs(pt.vdot(self.psi_target, self.psi_current))**2
        if new_fidelity > previous_fidelity:
            reward = 1  # Increased in fidelity
        elif new_fidelity < previous_fidelity:
            reward = -2  # penalty for decreased in fidelity
        else:
            reward = 0  # No change in fidelity
        
        self.psi_current = psi_next
        self.current_step += 1
        done = self.current_step >= self.max_steps or new_fidelity >= self.fidelity_threshold
 
        return psi_next, reward, done    
        
#%% 

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)
 

class DQNAgent:
    def __init__(self, action_space, state_dim, learning_rate, gamma, epsilon, update_target_every):
        self.action_space = action_space
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_target_every = update_target_every
        self.steps_done = 0
        
        self.model = DQNNetwork(state_dim, len(action_space))
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Make sure the target model's parameters are frozen
        for param in self.target_model.parameters():
            param.requires_grad = False
    
    def act(self, psi_current):
        if np.random.uniform(0, 1) < self.epsilon:             
            action_idx = np.random.randint(len(self.action_space))
        else:
            psi_current_Re_Im = pt.concatenate([psi_current.real, psi_current.imag], dim=-1)
            q_values = self.model(psi_current_Re_Im)
            action_idx = pt.argmax(q_values).item()
        return self.action_space[action_idx], action_idx
    
    def learn(self, psi_current, action_idx, reward, psi_next, done):
        psi_current_Re_Im = pt.unsqueeze(pt.concatenate([psi_current.real, psi_current.imag], dim=-1), 0)
        psi_next_Re_Im = pt.unsqueeze(pt.concatenate([psi_next.real, psi_next.imag], dim=-1), 0)
        
        reward = pt.FloatTensor([reward])
        done = pt.FloatTensor([done])
        
        current_q = self.model(psi_current_Re_Im).squeeze(0)[action_idx]
        with pt.no_grad():
            # Get the action with the highest value from the model for the next state
            next_actions = pt.argmax(self.model(psi_next_Re_Im).squeeze(0), dim=0)
            # Get the Q-value from the target model for the next action
            max_next_q = self.target_model(psi_next_Re_Im).squeeze(0)[next_actions]
        expected_q = reward + (1 - done) * self.gamma * max_next_q
        
        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            

def get_norm(psi):
    norm = pt.sum(pt.abs(psi)**2)
    return norm.item()



def generate_basis_via_sigma_z(L):
    D= 2**L
    basis = pt.zeros((2**L,L))
    for v in range(0,basis.shape[0]):
        fock_state = pt.zeros(D, dtype = pt.complex64)
        fock_state[v] = 1
        for i in range(1,L+1):
            basis[v,i-1] = pt.vdot(fock_state, Z[i]@fock_state)

    return (basis+1)/2



def basis_Fock_to_string(v_Fock):
    
    string = "|"
    for i in range(0,L):
        string = string + str(v_Fock[i])
    string = string + ">"
    
    return string

def print_quantum_state(psi, psi_string):
    print(psi_string)
    print("Re[|psi>]     Im[|psi>]")
    for i in range(psi.shape[0]):
        print("    {:2.2f}".format(psi[i].real) + "   + 1j*({:2.2f})".format(psi[i].imag))

    return

# Helper code helping defining target and final states in convinient notations
 
L = 3   # Number of qubits

D = 2**L
Id = get_chain_operator(id_local, L, 1)
Z = {}
for i in range(1,L+1):
    index = i
    Z[index] = get_chain_operator(sigma_z, L, i)

basis_Fock = generate_basis_via_sigma_z(L)
basis_Fock = np.array(basis_Fock.detach().numpy(), dtype = int)    
ket = {}
for v_idx in range(0,basis_Fock.shape[0]):
    v_Fock = basis_Fock[v_idx,:]
    v_Fock_string = basis_Fock_to_string(v_Fock)
    psi_tmp = pt.zeros((D,)) + 0*1j
    psi_tmp[v_idx] = 1
    ket[v_Fock_string] = psi_tmp
################################################################
 

 
psi_initial = ket["|000>"]                      # Initial state of quantum circuit
norm = pt.sum(pt.abs(psi_initial)**2)
psi_initial = psi_initial/pt.sqrt(norm)

psi_target = ket["|000>"] + ket["|111>"]                   # Target |GHZ> state
# psi_target = ket["|100>"] + ket["|010>"] + ket["|001>"]      # Target |W> state
norm = pt.sum(pt.abs(psi_target)**2)
psi_target = psi_target/pt.sqrt(norm)

fidelity_threshold  = 0.99                      # 
max_steps           = 20                        # maximum number of agent steps
learning_rate       = 1e-3                      # optimizer learning rate
gamma               = 0.99                      # discount factor
epsilon             = 0.3                       # epsilon-greedy strategy parameter
N_episodes          = 1000                      # number of training episodes


env = QuantumCircuitEnvironment(L, psi_initial, psi_target,  max_steps, fidelity_threshold)

 
update_target_every = 10  # Update target network every 10 steps
agent = DQNAgent(env.action_space, env.state_dimension, learning_rate, gamma, epsilon, update_target_every)


model = DQNNetwork(env.state_dimension, len(env.action_space))
 
update_target_every = 10  # Update target network every 10 steps
agent = DQNAgent(env.action_space, env.state_dimension, learning_rate, gamma, epsilon, update_target_every)

# Modified training loop
training_history = []
for episode in tqdm(range(N_episodes)):
    psi_current = env.reset()
    total_reward = 0
    done = False
    
    actions_hitory_at_episode = []
    while not done:
        action, action_idx = agent.act(psi_current)
        actions_hitory_at_episode.append(action)        # collect take actions
        psi_next, reward, done = env.step(action)
        agent.learn(psi_current, action_idx, reward, psi_next, done)
        psi_current = psi_next
        fidelity = pt.abs(pt.vdot(psi_target, env.psi_current))**2
        total_reward += reward
        
        
    
    dict_tmp = {
                "lr"                : learning_rate,
                "gamma"             : gamma,
                "epsilon"           : epsilon,
                "N_episodes"        : N_episodes,
                "fidelity_threshold": fidelity_threshold,
                "max_steps"         : max_steps,
                "episode"           : episode,
                "fidelity"          : fidelity.item(),
                "actions_history"   : actions_hitory_at_episode,
                "psi_target"        : psi_target,
                "psi_initial"       : psi_initial,
                "psi_final"         : env.psi_current,            
                }
    training_history.append(dict_tmp)
    if(fidelity > fidelity_threshold):
        break

training_history = pd.DataFrame(training_history)

#%%
episode_idx_optimal  = training_history['fidelity'].argmax()
set_of_gates_optimal = training_history['actions_history'][episode_idx_optimal]
fidelity_optimal     = training_history['fidelity'][episode_idx_optimal]
psi_final_optimal    = training_history['psi_final'][episode_idx_optimal]
fidelity_optimal     = training_history['fidelity'][episode_idx_optimal]

#%%
print("############################")
print("Optimal Episode :", episode_idx_optimal)
print("|psi_initial|^2 = ", pt.abs(psi_initial)**2, " | norm = " + "{:2.2f}".format(get_norm(psi_target)))
print("|psi_target|^2 = ", pt.abs(psi_target)**2, " | norm = " + "{:2.2f}".format(get_norm(psi_target)))
print("|psi_final|^2 = ", pt.abs(psi_final_optimal)**2, " | norm = " + "{:2.2f}".format(get_norm(psi_final_optimal)))
print("Fidelity |<psi_target|psi_final>|^2 = ", "{:2.2f}".format(fidelity_optimal))
print("Optimal set of gates:")
for layer, gate in enumerate(set_of_gates_optimal):
    print("layer: " + str(layer), gate)

print_quantum_state(psi_initial,"Initial state")
print_quantum_state(psi_target,"Target state")    
print_quantum_state(psi_final_optimal,"Final state")