import sys
import pickle
import os
import torch_geometric as ptg
import matplotlib.pyplot as plt
import networkx as nx
sys.path.insert(0,'..')

import utils
import visualization as vis

OUTPUT_DIR = "figures"
FILE_NAME_TEMPLATE = "{}_wind_graph.pickle"
GRAPH_TYPES = ("std", "mean")
NODE_SIZE = 7.
EPS = 0.01
FIGSIZE = 20
CMAP = "YlOrRd" # "plasma"

for graph_type in GRAPH_TYPES:
    file_name = FILE_NAME_TEMPLATE.format(graph_type)
    with open(file_name, "rb") as graph_file:
        graph = pickle.load(graph_file).to("cpu")

    max_pos = graph.pos.max(dim=0).values
    min_pos = graph.pos.min(dim=0).values
    ratio = (max_pos[1] - min_pos[1]).item()/2. # height / width (longer)

    fig = vis.plot_graph(graph, name="std", return_plot=True)

    graph_networkx = ptg.utils.to_networkx(graph, to_undirected=True)

    plot_pos = graph.pos.numpy()
    node_color = graph.x.flatten()

    fig, ax = plt.subplots(figsize=(FIGSIZE, ratio*FIGSIZE))
    plt.axis("scaled")
    ax.set_xlim(-1.-EPS, 1.+EPS)
    scaled_ylim = (1.+EPS)*ratio
    ax.set_ylim(-1.*scaled_ylim, scaled_ylim)
    plt.axis("off")

    nodes = nx.draw_networkx_nodes(graph_networkx, pos=plot_pos, node_size=NODE_SIZE,
            node_color=node_color, cmap=CMAP)
    nx.draw_networkx_edges(graph_networkx, pos=plot_pos, width=0.7)

    # Draw masked area
    for mask in graph.mask_limits:
        mask_width, mask_height = mask[1] - mask[0]
        #  ax.add_patch(plt.Rectangle(mask[0], mask_width, mask_height,
                            #  fill=True, linewidth=2, color="black", alpha=0.1))
        ax.add_patch(plt.Rectangle(mask[0], mask_width, mask_height,
                            fill=False, linewidth=5, color="red"))

    #plt.colorbar(nodes)

    plt.gca().set_position([0, 0, 1, 1])

    pdf_path = os.path.join(OUTPUT_DIR, "{}_example.pdf".format(graph_type))
    png_path = os.path.join(OUTPUT_DIR, "{}_example.png".format(graph_type))
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches = 0)
    plt.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    plt.close()

