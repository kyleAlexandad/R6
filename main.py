# main.py
import os
import sympy as sp
import torch
import torch.optim as optim
from state import State, CircuitNode, is_terminal
from network import AlphaZeroNet
from trainer import self_play_episode, update_network

# Hyperparameters.
MAX_ACTIONS = 50         
INPUT_SIZE = 2           
HIDDEN_SIZE = 64         
NUM_SIMULATIONS = 50     
MAX_DEPTH = 5            
NUM_EPISODES = 50        # Fine-tuning episodes for the new target.
PRETRAIN_CHECKPOINT = "pretrained_checkpoint.pth"

def main():
    user_input = input("Enter a polynomial (e.g., x**2 + y**2 + 2*x*y or x**2): ")
    try:
        target = sp.sympify(user_input)
    except Exception as e:
        print("Error parsing polynomial:", e)
        return
    target = sp.simplify(target)
    print("Target polynomial:", target)
    
    free_symbols = list(target.free_symbols)
    if len(free_symbols) == 0:
        print("Target is a constant:", target)
        return
    print("Free symbols in target:", free_symbols)
    
    # For a trivial target, return immediately.
    if len(target.free_symbols) == 1 and target == free_symbols[0]:
        print("Trivial target detected. Circuit is simply:")
        print(target)
        return
    
    # Allowed creation actions: exactly the free symbols present.
    allowed_creations = free_symbols.copy()
    coeff, _ = target.as_coeff_add()
    if coeff != 0:
        allowed_creations.extend([1, 2, 3])
    print("Allowed creation actions:", allowed_creations)
    
    # Initial state is empty.
    initial_state = State([], [])
    
    net = AlphaZeroNet(INPUT_SIZE, HIDDEN_SIZE, MAX_ACTIONS)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Load the pretrained model if it exists.
    if os.path.exists(PRETRAIN_CHECKPOINT):
        checkpoint = torch.load(PRETRAIN_CHECKPOINT)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded pretrained model from", PRETRAIN_CHECKPOINT)
    else:
        print("No pretrained model found. Starting from scratch.")
    
    # Optionally, fine-tune the model on the new target.
    start_episode = 0
    for episode in range(start_episode, NUM_EPISODES):
        trajectory, reward, final_state = self_play_episode(
            initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH, allowed_creations
        )
        loss = update_network(net, optimizer, trajectory, reward)
        print(f"Episode {episode+1}/{NUM_EPISODES}: Reward = {reward:.2f}, Loss = {loss:.2f}")
        if is_terminal(final_state, target):
            print("Found valid circuit in episode", episode+1)
            print("Circuit tree:")
            final_state.nodes[0].print_tree()
            break
    
    # Save the fine-tuned model if desired.
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "finetuned_checkpoint.pth")
    
    print("\nFinal test:")
    trajectory, reward, final_state = self_play_episode(
        initial_state, target, net, MAX_ACTIONS, NUM_SIMULATIONS, MAX_DEPTH, allowed_creations
    )
    if is_terminal(final_state, target):
        print("Found a valid circuit!")
        final_state.nodes[0].print_tree()
    else:
        print("Did not find a valid circuit.")
    print("Final reward:", reward)

if __name__ == "__main__":
    main()
