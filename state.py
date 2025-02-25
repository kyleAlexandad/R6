# state.py
import sympy as sp

class CircuitNode:
    """
    Represents a node in the arithmetic circuit.
    - For leaves: operator is None and expr holds a variable or constant.
    - For internal nodes: operator is 'add' or 'mul' and left/right are child nodes.
    """
    def __init__(self, expr, operator=None, left=None, right=None):
        self.expr = sp.simplify(expr)
        self.operator = operator    # 'add' or 'mul' for internal nodes; None for leaves.
        self.left = left            # Left child (CircuitNode) if internal.
        self.right = right          # Right child (CircuitNode) if internal.
    
    def is_leaf(self):
        return self.operator is None
    
    def __str__(self):
        if self.is_leaf():
            return str(self.expr)
        else:
            return f"({str(self.left)} {self.operator} {str(self.right)})"
    
    def print_tree(self, indent=0):
        space = " " * indent
        if self.is_leaf():
            print(space + str(self.expr))
        else:
            print(space + self.operator.upper() + " gate")
            self.left.print_tree(indent + 4)
            self.right.print_tree(indent + 4)

class State:
    """
    Represents the current state of the circuit synthesis process.
    Attributes:
      - nodes: list of CircuitNode objects currently available.
      - history: list of actions taken to reach this state.
    """
    def __init__(self, nodes, history):
        self.nodes = nodes          # List of CircuitNode objects.
        self.history = history      # List of actions (tuples).

def get_possible_actions(state, allowed_creations):
    """
    Returns a list of valid actions.
    Two types of actions:
      1. Creation actions: ("create", item) for each allowed creation item.
      2. Combination actions: if there are at least 2 nodes, ("add", i, j) and ("mul", i, j).
    """
    actions = []
    # Creation actions are always allowed.
    for item in allowed_creations:
        actions.append(("create", item))
    n = len(state.nodes)
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                actions.append(("add", i, j))
                actions.append(("mul", i, j))
    return actions

def apply_action(state, action):
    """
    Applies an action to the state and returns a new state.
    - For a creation action ("create", item): a new node for that symbol is added.
    - For a combination action ("add", i, j) or ("mul", i, j): combine the two nodes.
    """
    if action[0] == "create":
        # Create a new node from the given item.
        item = action[1]
        new_node = CircuitNode(item)
        new_nodes = state.nodes.copy()
        new_nodes.append(new_node)
        new_history = state.history + [action]
        return State(new_nodes, new_history)
    else:
        op, i, j = action
        node_a = state.nodes[i]
        node_b = state.nodes[j]
        if op == 'add':
            new_expr = sp.simplify(node_a.expr + node_b.expr)
        elif op == 'mul':
            new_expr = sp.simplify(node_a.expr * node_b.expr)
        else:
            raise ValueError("Unknown operation")
        new_node = CircuitNode(new_expr, operator=op, left=node_a, right=node_b)
        new_nodes = state.nodes.copy()
        # Remove the two nodes (in descending order) and add the new node.
        for index in sorted([i, j], reverse=True):
            new_nodes.pop(index)
        new_nodes.append(new_node)
        new_history = state.history + [action]
        return State(new_nodes, new_history)

def is_terminal(state, target):
    """
    A state is terminal if there is exactly one node and its expression equals the target (symbolically).
    """
    if len(state.nodes) == 1:
        return sp.simplify(state.nodes[0].expr - target) == 0
    return False

def compute_reward(state, target):
    """
    Computes the reward for a state.
    If terminal, reward = 100 minus the number of operations.
    Otherwise, a heuristic reward is given based on the similarity of one node to the target.
    Here, similarity is measured simply by the difference in string lengths.
    """
    if is_terminal(state, target):
        return 100 - len(state.history)
    best_sim = 0
    for node in state.nodes:
        if sp.simplify(node.expr - target) == 0:
            return 100 - len(state.history)
        diff = abs(len(str(node.expr)) - len(str(target)))
        sim = 1.0 / (1.0 + diff)
        if sim > best_sim:
            best_sim = sim
    return (100 - len(state.history)) * best_sim
