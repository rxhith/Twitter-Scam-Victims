import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON data
with open('network.json', 'r') as json_file:
    data = json.load(json_file)

# Create a NetworkX graph
G = nx.Graph()

# Iterate through the data and add nodes and edges
for node, info in data.items():
    connections = info['Connections']

    # Add the node to the graph if it doesn't exist
    if node not in G.nodes:
        G.add_node(node)

    # Add edges to the graph
    for neighbor in connections:
        # Check if the edge already exists and is not a self-loop
        if node != str(neighbor):
            G.add_edge(node, str(neighbor))

# Remove self-loops from the graph
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from(self_loops)

# Plot the NetworkX graph
pos = nx.spring_layout(G)  # Layout the nodes
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10)
plt.title('NetworkX Graph Based on Connections (Self-loops removed)')
plt.show()

# Calculate the total number of nodes in the graph
total_nodes = G.number_of_nodes()
print(f'Total number of nodes in the graph: {total_nodes}')
