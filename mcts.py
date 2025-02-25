# mcts.py
import math
import random
import torch
from state import get_possible_actions, apply_action, is_terminal, compute_reward
from network import encode_state

CPUCT = 1.0

class Node:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state            # A State object (might be None until expanded).
        self.parent = parent
        self.action_taken = action_taken  # The action taken to reach this node.
        self.children = {}            # Mapping: action -> Node.
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0              # Prior probability from the network.

    def q_value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

    def uct_score(self, child):
        return child.q_value() + CPUCT * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

def select_child(node, valid_actions):
    # Only consider actions that are already expanded (i.e. keys in node.children)
    available_actions = [action for action in valid_actions if action in node.children]
    if not available_actions:
        # Fallback: if none of the truncated valid_actions were expanded, return a random one from the truncated list.
        return valid_actions[0]
    best_score = -float("inf")
    best_action = None
    for action in available_actions:
        score = node.uct_score(node.children[action])
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

def expand_node(node, valid_actions, net, max_actions):
    # Truncate valid actions to at most max_actions.
    valid_actions_limited = valid_actions[:max_actions]
    N = len(valid_actions_limited)
    state_vec = encode_state(node.state)
    policy_logits, _ = net(state_vec)
    # Create a mask so that only the first N logits are used.
    mask = torch.full((max_actions,), -float("inf"))
    mask[:N] = 0
    masked_logits = policy_logits + mask
    probs = torch.softmax(masked_logits, dim=0).detach().numpy()
    
    for idx, action in enumerate(valid_actions_limited):
        if action not in node.children:
            child = Node(state=None, parent=node, action_taken=action)
            child.prior = probs[idx]
            node.children[action] = child

def simulate(node, target, net, max_actions, depth, max_depth, allowed_creations):
    if node.state is None:
        node.state = apply_action(node.parent.state, node.action_taken)
    if is_terminal(node.state, target):
        return compute_reward(node.state, target)
    if depth >= max_depth:
        state_vec = encode_state(node.state)
        _, value = net(state_vec)
        return value.item()
    # Get all valid actions, then truncate to at most max_actions.
    valid_actions_full = get_possible_actions(node.state, allowed_creations)
    valid_actions = valid_actions_full[:max_actions]
    if not valid_actions:
        return 0
    if len(node.children) == 0:
        expand_node(node, valid_actions, net, max_actions)
        state_vec = encode_state(node.state)
        _, value = net(state_vec)
        return value.item()
    action = select_child(node, valid_actions)
    # If select_child somehow returns an action not in children, fallback:
    if action not in node.children:
        action = list(node.children.keys())[0]
    child = node.children[action]
    value = simulate(child, target, net, max_actions, depth + 1, max_depth, allowed_creations)
    child.visit_count += 1
    child.total_value += value
    node.visit_count += 1
    return value

def run_mcts(root_state, target, net, max_actions, num_simulations, max_depth, allowed_creations):
    root = Node(root_state)
    for _ in range(num_simulations):
        simulate(root, target, net, max_actions, depth=0, max_depth=max_depth, allowed_creations=allowed_creations)
    valid_actions_full = get_possible_actions(root.state, allowed_creations)
    valid_actions = valid_actions_full[:max_actions]
    best_action = None
    best_visits = -1
    for action in valid_actions:
        if action in root.children:
            visits = root.children[action].visit_count
            if visits > best_visits:
                best_visits = visits
                best_action = action
    return best_action, root
