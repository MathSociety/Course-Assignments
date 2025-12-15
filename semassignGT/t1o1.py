class Node:
    def __init__(self, node_name, player=None, payoff=None):
        self.node_name = node_name
        self.player = player
        self.node_branches = []  # List of (branch_name, child_node) tuples
        self.payoff = payoff  # Only for leaf nodes

    def add_branch(self, branch_name, child_node):
        self.node_branches.append((branch_name, child_node))

    def __repr__(self, level=0):
        indent = "  " * level
        result = f"{indent}Node({self.node_name}, Player: {self.player}, Payoff: {self.payoff})\n"
        for branch_name, child in self.node_branches:
            result += f"{indent}  Branch: {branch_name} ->\n"
            result += child.__repr__(level + 2)
        return result


def build_tree(piefg, root_name="V0"):
    def process_node(node_name):
        """Process a node and return a corresponding Node object."""
        for entry in piefg:
            if isinstance(entry, list) and node_name in entry:
                # Leaf node with payoff
                if isinstance(entry[-1], list):
                    return Node(node_name, payoff=tuple(entry[-1]))
                # Decision node with branches
                elif isinstance(entry[-1], dict):
                    player = entry[1]
                    node = Node(node_name, player=player)
                    for action, child_name in entry[-1].items():
                        child_node = process_node(child_name)
                        node.add_branch(action, child_node)
                    return node
        return None

    # Build the tree starting from the root
    return process_node(root_name)


# Example PIEF input
piefg = [
    ['P1', 'P2'],
    'V0',
    ['V0', 'P1', {'V1': 'Stop', 'V2': 'Go'}],
    ['V1', 'P2', {'V3': 'Ready', 'V4': 'Go', 'V5': 'Start'}],
    ['V3', [3, 8]],
    ['V4', [8, 10]],
    ['V5', [6, 3]]
]

# Build the tree
tree_root = build_tree(piefg)

# Print the tree structure
print(tree_root)
