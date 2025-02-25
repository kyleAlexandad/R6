# pretrain.py
import os
import sympy as sp
import torch
import torch.optim as optim
from state import State, CircuitNode, is_terminal
from network import AlphaZeroNet
from trainer import self_play_episode, update_network

# List of training polynomials (unsimplified or diverse forms)
# You can expand this list as needed.
polynomials = [
    "x**2 + y**2 + 2*x*y",
    "x**2 + 3*x*y + y**2",
    "x**3 + 2*x**2*y + x*y**2",
    "x**2 - y**2",
    "x**2 + y**2",
    "x**3 + y**3",
    "x**2 + 2*x + 1",
    "y**2 - 2*y + 1",
    "x**2 + 2*y**2",
    "2*x**2 + 3*y**2",
    "x*y + x + y",
    "x**2 + x*y + y**2",
    "x**3 + y",
    "x**2*y + y**2",
    "x**2 + y**3",
    "x**3 + y**3",
    "x**4 + y**4",
    "x**2 - 2*x*y + y**2",
    "2*x**2 + 4*x*y + 2*y**2",
    "x**2 + 2*x*y",  # deliberately unsimplified
]

# Duplicate the list to ensure a good variety (here 20 * 3 = 60 examples)
training_polynomials = polynomials * 3

# Hyperparameters for pretraining.
MAX_ACTIONS = 50         
INPUT_SIZE = 2           
HIDDEN_SIZE = 64         
NUM_SIMULATIONS = 50     
MAX_DEPTH = 5            
NUM_EPISODES_PER_POLY = 10  # Episodes per polynomial during pretraining.
PRETRAIN_CHECKPOINT = "pretrained_checkpoint.pth"

def pretrain():
    net = AlphaZeroNet(INPUT_SIZE, HIDDEN_SIZE, MAX_ACTIONS)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Iterate over each training polynomial.
    for poly_str in training_polynomials:
        try:
            target = sp.sympify(poly_str)
        except Exception as e:
            print(f"Error parsing polynomial {poly_str}: {e}")
            continue
        target = sp.simplify(target)
        free_symbols = list(target.free_symbols)
        if len(free_symbols) == 0:
            continue
        # Allowed creation actions: only the free symbols present.
        allowed_creations = free_symbols.copy()
        coeff, _ = target.as_coeff_add()
        if coeff != 0:
            allowed_creations.extend([1, 2, 3])
        
        # Use an empty initial state.
        initial_state = State([], [])
        print(f"\nPretraining on polynomial: {target}")
        for episode in range(NUM_EPISODES_PER_POLY):
            trajectory, reward, final_state = self_play_episode(
                initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH, allowed_creations
            )
            loss = update_network(net, optimizer, trajectory, reward)
            print(f"  Episode {episode+1}/{NUM_EPISODES_PER_POLY}: Reward = {reward:.2f}, Loss = {loss:.2f}")
    
    # Save the pretrained model checkpoint.
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, PRETRAIN_CHECKPOINT)
    print("\nPretraining complete. Saved pretrained model to", PRETRAIN_CHECKPOINT)

if __name__ == "__main__":
    pretrain()
