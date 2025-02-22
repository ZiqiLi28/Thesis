import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
g = nx.DiGraph()

# Add nodes
g.add_node("Frontend (React)", pos=(0, 1.5), description="User Interface for image upload and result display")
g.add_node("Backend (Flask)", pos=(0, 1), description="Handles API requests, integrates models")
g.add_node("EasyOCR", pos=(-3, 0), description="Extracts text from detected plates")
g.add_node("YOLO", pos=(-1, 0), description="Detects license plates in image")
g.add_node("Azure Custom Vision", pos=(1, 0), description="Detects license plates in image")
g.add_node("Azure Computer Vision", pos=(3, 0), description="Extracts text from detected plates")

# Add edges with labels
edges_with_labels = {
    ("Frontend (React)", "Backend (Flask)"): "",
    ("Backend (Flask)", "YOLO"): "Image",
    ("YOLO", "EasyOCR"): "License plate",
    ("EasyOCR", "Backend (Flask)"): "Send text",
    ("Backend (Flask)", "Azure Custom Vision"): "Image",
    ("Azure Custom Vision", "Azure Computer Vision"): "License plate",
    ("Azure Computer Vision", "Backend (Flask)"): "Send text",
    ("Backend (Flask)", "Frontend (React)"): ""
}

# Add edges to the graph
g.add_edges_from(edges_with_labels.keys())

# Get node positions
pos = nx.get_node_attributes(g, 'pos')

plt.figure(figsize=(12, 8))

# Draw arrows first (background layer, gray)
nx.draw_networkx_edges(
    g, pos, edge_color="gray", width=2, arrows=True, arrowsize=12
)

# Draw nodes and labels (foreground)
nx.draw(
    g, pos, with_labels=True, node_color="lightblue", node_size=5000,
    font_size=15, font_weight="bold", edge_color="black"
)

# Add descriptions below nodes
descriptions = nx.get_node_attributes(g, 'description')
for node, (x, y) in pos.items():
    plt.text(x, y - 0.15, descriptions[node], fontsize=15, ha='center', wrap=True)

# Add edge labels
nx.draw_networkx_edge_labels(
    g, pos, edge_labels=edges_with_labels, font_size=12, label_pos=0.5
)

plt.title("System Architecture for License Plate Recognition", fontsize=16)
plt.show()
