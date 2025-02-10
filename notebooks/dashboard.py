import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import jinja2
import numpy as np
import random
import matplotlib.pyplot as plt
from simulation import simulate_turn_based_games
from community_analysis import compute_community_detection, semantic_cluster, compute_bridge_nodes, perturbation_analysis, community_centric_strategies, cultural_economic_correlations

country_graph = nx.read_graphml("country_graph.graphml")
G_undirected = country_graph.to_undirected()
cities_df = pd.read_csv("../data/cities.csv")

# Use new caching API (st.cache_data)
@st.cache_data
def load_graph(graph_type):
    # Load graph based on user selection
    if graph_type == "Country":
        return nx.read_graphml("country_graph.graphml")
    elif graph_type == "City":
        return nx.read_graphml("city_graph.graphml")
    elif graph_type == "Combined":
        return nx.read_graphml("combined_graph.graphml")
    else:
        return nx.read_graphml("country_graph.graphml")

@st.cache_data
def load_countries_graph():
    G = nx.read_graphml("country_graph.graphml")
    return G, G.to_undirected()

@st.cache_data
def load_cities_df():
    return pd.read_csv("../data/cities.csv")

def pyvis_visualization(graph):
    net = Network(height="100vh", width="100vw", bgcolor="#222222", font_color="white")
    net.from_nx(graph)
    template_loader = jinja2.PackageLoader('pyvis', 'templates')
    template_env = jinja2.Environment(loader=template_loader)
    net.template = template_env.get_template("template.html")
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    html_file = "temp_graph.html"
    net.show(html_file)
    return html_file

def community_visualization(graph, partition=None):
    net = Network(height="100vh", width="100vw", bgcolor="#222222", font_color="white")
    net.from_nx(graph)
    
    # If a community partition is provided, assign each node a color based on its community.
    if partition is not None:
        community_ids = list(set(partition.values()))
        # Generate a color for each community (using random hex colors)
        colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(community_ids))]
        for node in net.nodes:
            node_id = node.get('id')
            if node_id in partition:
                node['color'] = colors[ partition[node_id] % len(colors) ]
    
    template_loader = jinja2.PackageLoader('pyvis', 'templates')
    template_env = jinja2.Environment(loader=template_loader)
    net.template = template_env.get_template("template.html")
    
    # Use modified physics settings to stabilize the network (reduce movement)
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 200, "updateInterval": 25},
        "minVelocity": 0.1,
        "maxVelocity": 0.5,
        "forceAtlas2Based": {
          "gravitationalConstant": -30,
          "centralGravity": 0.02,
          "springLength": 100,
          "springConstant": 0.05
        }
      }
    }
    """)
    
    html_file = "temp_graph.html"
    net.show(html_file)
    return html_file


def compute_additional_metrics(graph):
    degree_centrality = nx.degree_centrality(graph)
    # Use out_degree if the graph is directed; otherwise, use degree for undirected graphs.
    if graph.is_directed():
        out_degrees = {n: graph.out_degree(n) for n in graph.nodes()}
    else:
        out_degrees = {n: graph.degree(n) for n in graph.nodes()}
    # Dummy trap score: inverse of the out-degree (or degree) plus one.
    trap_score = {n: (1 / (out_degrees[n] + 1)) for n in graph.nodes()}
    return degree_centrality, out_degrees, trap_score


# def simulate_single_strategy_game(graph, strategy_func, start_node, start_counts):
#     """
#     Simulate a game starting from 'start_node' using a single strategy (strategy_func).
#     Returns the length of the game (number of moves until no valid moves remain).
#     """
#     current = start_node
#     visited = {current}
#     path_length = 0
#     while True:
#         moves = [n for n in graph.successors(current) if n not in visited]
#         if not moves:
#             break
#         # Choose the move based on the provided strategy function
#         move = strategy_func(graph, current, visited, start_counts)
#         if move is None:
#             break
#         current = move
#         visited.add(current)
#         path_length += 1
#     return path_length


def main():
    st.title("Atlas Game Graph Dashboard")

    # Graph selection
    graph_type = st.radio("Select Graph:", ("Country", "City", "Combined"))
    graph = load_graph(graph_type)
    
    st.subheader("Interactive Graph Visualization")
    html_file = pyvis_visualization(graph)
    st.components.v1.html(open(html_file, 'r', encoding='utf-8').read(), height=600)
    
    # Display additional metrics for a selected node
    degree_centrality, out_degrees, trap_score = compute_additional_metrics(graph)
    
    st.subheader("Explore Node Metrics")
    selected_node = st.text_input("Enter a node (e.g., AFGHANISTAN):", value="AFGHANISTAN").upper()
    if selected_node in graph.nodes():
        # Check if the graph is directed; if not, use degree
        if graph.is_directed():
            edge_count = graph.out_degree(selected_node)
        else:
            edge_count = graph.degree(selected_node)
        st.write("Outgoing Edges Count:", edge_count)
        st.write("Degree Centrality:", degree_centrality.get(selected_node, "N/A"))
        st.write("Trap Score:", trap_score.get(selected_node, "N/A"))
    else:
        st.write("Node not found in the graph.")

        # Survival path length simulation (with adjustable iterations)
    #     iterations = st.slider("Monte Carlo iterations for survival simulation:", 100, 2000, 500)

    #     def simulate_survival(graph, node, iterations):
    #         lengths = []
    #         for _ in range(iterations):
    #             lengths.append(simulate_single_strategy_game(graph, random, node, 0))
    #         return np.mean(lengths)
    #     survival_length = simulate_survival(graph, selected_node, iterations)
    #     st.write("Estimated Survival Path Length:", survival_length)
    # else:
    #     st.write("Node not found in the graph.")
    
    # Simulation of game strategies
    st.subheader("Simulate Turn-Based Game Strategies")
    num_games = st.number_input("Number of games to simulate:", min_value=10, max_value=1000, value=100, step=10)

    # When the user clicks the "Run Simulation" button, run the simulation on the currently loaded graph.
    if st.button("Run Simulation"):
        # We'll capture the output printed by simulate_turn_based_games and display it in the dashboard.
        import io
        import contextlib
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            simulate_turn_based_games(graph, num_games=int(num_games), verbose=True)
        simulation_results = output_buffer.getvalue()
        st.text(simulation_results)
    
    # Letter Frequency Analysis with sorted order and labels
    st.subheader("Letter Frequency Analysis")
    nodes = list(graph.nodes())
    start_letters = [n[0] for n in nodes if n]
    end_letters = [n[-1] for n in nodes if n]
    start_freq = pd.Series(start_letters).value_counts().sort_values(ascending=False)
    end_freq = pd.Series(end_letters).value_counts().sort_values(ascending=False)

    st.write("Starting Letter Frequency")
    st.bar_chart(start_freq)
    st.write("Ending Letter Frequency")
    st.bar_chart(end_freq)

    
    # Survival Path Visualization for a node using nx.algorithms.dag_longest_path (if DAG)
    st.subheader("Longest Path Visualization")
    if st.button("Show Longest Path for Selected Node"):
        # Use DFS with a maximum depth to prevent long runtimes
        max_depth = 7  # adjustable parameter
        def dfs_longest_path(graph, current, visited, depth):
            if depth >= max_depth:
                return [current]
            longest = [current]
            for neighbor in list(graph.successors(current)):
                if neighbor not in visited:
                    path = dfs_longest_path(graph, neighbor, visited | {neighbor}, depth + 1)
                    if len(path) + 1 > len(longest):
                        longest = [current] + path
            return longest
        if selected_node in graph.nodes():
            longest = dfs_longest_path(graph, selected_node, {selected_node}, 0)
            st.write("Longest Path (limited depth):", longest)
            subG = graph.subgraph(longest)
            pos = nx.spring_layout(subG)
            fig_lp, ax_lp = plt.subplots(figsize=(8, 6))
            nx.draw(subG, pos, with_labels=True, node_color='skyblue', edge_color='gray', ax=ax_lp)
            ax_lp.set_title("Longest Path from " + selected_node)
            st.pyplot(fig_lp)
        else:
            st.write("Node not found in the graph.")


    st.subheader("Community Detection Analysis")
    community_method = st.radio("Select Community Visualization Method:", ["Louvain", "Girvan-Newman"])
    # Reuse the community detection results computed earlier
    # (Ensure these variables are available by calling load_countries_graph() and compute_community_detection)
    country_graph, G_undirected = load_countries_graph()
    community_results = compute_community_detection(country_graph)
    partition_louvain = community_results["louvain_partition"]
    partition_gn = community_results["girvan_partition"]

    if community_method == "Louvain":
        st.write("Displaying Louvain Community Graph")
        html_comm = community_visualization(G_undirected, partition=partition_louvain)
    else:
        st.write("Displaying Girvan-Newman Community Graph")
        html_comm = community_visualization(G_undirected, partition=partition_gn)
        
    st.components.v1.html(open(html_comm, 'r', encoding='utf-8').read(), height=600)

    country_graph, G_undirected = load_countries_graph()
    cities_df = load_cities_df()
    community_results = compute_community_detection(country_graph)
    partition_louvain = community_results["louvain_partition"]
    modularity_louvain = community_results["modularity_louvain"]
    communities_louvain = community_results["communities_louvain"]
    partition_gn = community_results["girvan_partition"]
    modularity_gn = community_results["modularity_gn"]
    communities_gn = community_results["communities_gn"]

    st.write(f"**Louvain Method:** {len(communities_louvain)} communities, Modularity: {modularity_louvain:.4f}")
    st.write(f"**Girvan-Newman Method:** {len(communities_gn)} communities, Best Modularity: {modularity_gn:.4f}")

    # Display community size distribution as a bar chart
    louvain_sizes = {comm: len(nodes) for comm, nodes in communities_louvain.items()}
    gn_sizes = {comm: len(nodes) for comm, nodes in communities_gn.items()}
    df_louvain_sizes = pd.DataFrame(list(louvain_sizes.items()), columns=["Community", "Size"]).sort_values(by="Size", ascending=False)
    df_gn_sizes = pd.DataFrame(list(gn_sizes.items()), columns=["Community", "Size"]).sort_values(by="Size", ascending=False)
    st.write("**Louvain Community Size Distribution:**")
    st.bar_chart(df_louvain_sizes.set_index("Community"))
    st.write("**Girvan-Newman Community Size Distribution:**")
    st.bar_chart(df_gn_sizes.set_index("Community"))

    # Semantic Community Analysis
    semantic_clusters = semantic_cluster(list(G_undirected.nodes()))
    st.write("**Semantic Clusters:**", {k: len(v) for k, v in semantic_clusters.items()})

    # Bridge Node Analysis
    bridge_nodes, top_bridges = compute_bridge_nodes(G_undirected, threshold=0.05)
    st.write("**Bridge Nodes (threshold 0.05):**", bridge_nodes)
    st.write("**Top 10 Bridge Nodes by Betweenness:**", top_bridges)

    # Perturbation Analysis
    mod_perturbed = perturbation_analysis(G_undirected, removal_fraction=0.05)
    st.write(f"**Original Louvain Modularity:** {modularity_louvain:.4f}")
    st.write(f"**Perturbed Louvain Modularity (5% edges removed):** {mod_perturbed:.4f}")

    # Community-Centric Strategies: Internal Density
    density_info = community_centric_strategies(G_undirected, partition_louvain)
    st.write("**Community Internal Densities (Louvain):**", density_info)

    # Cultural/Economic Correlations (if countries_df has population)
    pop_by_comm = cultural_economic_correlations(cities_df, partition_louvain, name_column='city', pop_column='population')
    st.write("**Average Population by Community (Louvain, from cities_df):**", pop_by_comm)

if __name__ == "__main__":
    main()
