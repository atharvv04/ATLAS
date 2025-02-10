import networkx as nx
import community as louvain_community
from networkx.algorithms.community import girvan_newman, modularity
import random
import pandas as pd

def compute_community_detection(G):
    # Convert graph to undirected
    G_undirected = G.to_undirected()
    
    # --- Louvain Method ---
    partition_louvain = louvain_community.best_partition(G_undirected)
    communities_louvain = {}
    for node, comm_id in partition_louvain.items():
        communities_louvain.setdefault(comm_id, []).append(node)
    modularity_louvain = louvain_community.modularity(partition_louvain, G_undirected)
    
    # --- Girvan-Newman Method ---
    gn_generator = girvan_newman(G_undirected)
    best_partition_gn = None
    max_modularity_gn = -1
    max_iter = 15  # Allow up to 15 iterations
    for i, partition in enumerate(gn_generator):
        if i >= max_iter:
            break
        current_mod = modularity(G_undirected, partition)
        if current_mod > max_modularity_gn:
            max_modularity_gn = current_mod
            best_partition_gn = partition
    partition_gn = {}
    for comm_id, comm in enumerate(best_partition_gn):
        for node in comm:
            partition_gn[node] = comm_id
            
    return {
        "louvain_partition": partition_louvain,
        "communities_louvain": communities_louvain,
        "modularity_louvain": modularity_louvain,
        "girvan_partition": partition_gn,
        "communities_gn": {i: list(comm) for i, comm in enumerate(best_partition_gn)},
        "modularity_gn": max_modularity_gn
    }

def semantic_cluster(names):
    """
    Cluster country names semantically based on common linguistic suffixes.
    This version checks case-insensitively.
    """
    clusters = {"-stan": [], "-land": [], "-ia": [], "others": []}
    for name in names:
        lower = name.lower()
        if lower.endswith("stan"):
            clusters["-stan"].append(name)
        elif lower.endswith("land"):
            clusters["-land"].append(name)
        elif lower.endswith("ia"):
            clusters["-ia"].append(name)
        else:
            clusters["others"].append(name)
    return clusters

def compute_bridge_nodes(G, threshold=0.05):
    """
    Compute betweenness centrality and return a list of nodes whose betweenness is above the threshold,
    as well as the top 10 nodes by betweenness.
    """
    betweenness = nx.betweenness_centrality(G)
    bridge_nodes = [node for node, val in betweenness.items() if val > threshold]
    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    return bridge_nodes, top_bridges

def perturbation_analysis(G, removal_fraction=0.05):
    """
    Remove a fraction of edges and recompute Louvain modularity.
    Returns the perturbed modularity.
    """
    G_perturbed = G.copy()
    edges_to_remove = random.sample(list(G.edges()), int(removal_fraction * G.number_of_edges()))
    G_perturbed.remove_edges_from(edges_to_remove)
    partition_perturbed = louvain_community.best_partition(G_perturbed)
    mod_perturbed = louvain_community.modularity(partition_perturbed, G_perturbed)
    return mod_perturbed

def community_centric_strategies(G, partition):
    """
    For each community, compute the internal density (ratio of internal edges to maximum possible edges).
    Returns a dictionary mapping community id to its internal density.
    """
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)
    density_info = {}
    for comm, nodes in communities.items():
        subG = G.subgraph(nodes)
        n = len(nodes)
        if n > 1:
            max_possible = n * (n - 1) / 2
            actual = subG.number_of_edges()
            density_info[comm] = actual / max_possible
        else:
            density_info[comm] = 0
    return density_info

def cultural_economic_correlations(df, partition, name_column='Name', pop_column='population'):
    """
    Enrich node data with community assignments and compute average population per community.
    If the DataFrame doesn't have a 'Name' column, it falls back to using 'city'.
    Expects a column for population specified by pop_column.
    """
    if name_column not in df.columns:
        if 'city' in df.columns:
            name_column = 'city'
        else:
            raise KeyError("DataFrame must contain a 'Name' or 'city' column.")
    
    # Ensure population column exists
    if pop_column not in df.columns:
        raise KeyError(f"DataFrame must contain a '{pop_column}' column for population data.")
    
    df = df.copy()
    df['Name_upper'] = df[name_column].str.upper()
    df['Community'] = df['Name_upper'].apply(lambda x: partition.get(x.lower(), -1))
    pop_by_comm = df.groupby('Community')[pop_column].mean().to_dict()
    return pop_by_comm

