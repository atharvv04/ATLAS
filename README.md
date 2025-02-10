# 1. Introduction and Objectives

The Atlas game is a turn‑based word game where players alternate naming places (countries, cities, or states) such that each new name starts with the last letter of the previous name. For example, if a player says “India,” the opponent must name a place beginning with “A” (e.g., “Australia”). A move that repeats a name or fails to produce a valid response results in elimination or a loss of points. 

Our objective is to apply graph theory to the Atlas game in order to derive strategic insights that may give players a competitive advantage. By representing places as nodes and drawing a directed edge from node A to node B when the last letter of A equals the first letter of B (the “Atlas rule”), we construct a network that captures the connectivity patterns inherent in the game. NetworkX is used for graph construction, and multiple datasets are leveraged:  
- **Countries:** 249 unique country names (sourced from the ISO 3166-1 country list via [DataHub Country List](https://datahub.io/core/country-list)).  
- **Cities:** Approximately 500 unique city names (sourced from [World Population Review](https://worldpopulationreview.com/cities)), where duplicates have been removed.

# 1. Methodology Overview
### **Graph Construction:**
Using the Atlas rule—“an edge from A to B if the last letter of A equals the first letter of B”—we construct directed graphs for countries, cities, and a combined dataset. The graphs are built using NetworkX, where each node represents a place, and edges capture valid transitions according to the game’s rule. This process transforms linguistic patterns into a structured network that can be analyzed using standard graph theory methods.

### **Metrics Computed:**  
  We calculate standard network metrics including degree centrality, betweenness centrality, and then a custom “Trap Score” (defined as the inverse of the node’s out‑degree plus one). We also run a Monte Carlo simulation to estimate the survival path length for nodes.

## **Simulation of Game Strategies:**  
- Multiple strategies are compared:
  - **Random Strategy:** At each move, a random valid successor is chosen.
  - **Letter Gap Strategy:** At each move, the strategy chooses the valid move whose last letter is rare (i.e. minimizes the count of nodes starting with that letter in the full graph).
  - **Composite and Look-Ahead Strategies (Exploratory):**  
  Alternative strategies that combine multiple criteria (e.g., out‑degree and trap score) or perform a one‑step look-ahead were also explored.

- **Simulation Results:**  
  - Simulations over multiple iterations revealed significant differences. In one instance, the Letter Gap Strategy produced a win rate of 96% compared to only 4% for a pure random strategy. 
  - However, other simulations with different strategic formulations (e.g., using pure out‑degree maximization) yielded less favorable results. This suggests that the Atlas network is highly sensitive to the specific decision rule used, and that selecting moves based on rare starting letters can be particularly advantageous.

# **2. Visualizations:**  
### 2D and 3D Network Layouts

- **2D Visualizations:**  
  We generated interactive 2D visualizations using Plotly and Matplotlib. These plots, based on the Kamada‑Kawai layout, reveal how nodes (countries/cities) are connected and help identify clusters. For example, strategic nodes such as Afghanistan, Azerbaijan, and Argentina appear as hubs in the 2D layout.
  
- **3D Visualizations:**  
  A 3D animated Plotly visualization was implemented to display the network on a “globe‑like” structure. This dynamic view enhances our understanding of spatial clusters and connectivity patterns that are not as apparent in 2D.

- **PyVis Interactive Visualization:**  
  The PyVis network visualization opens in a browser and allows users to interact with the graph (e.g., zoom, drag nodes, and hover to see node labels). Adjustments to physics parameters (e.g., stabilization and lower velocity) ensure that nodes do not move excessively, providing a more stable view of the network.

### Interactive Dashboard

We developed a Streamlit dashboard that integrates all the visualizations and analyses:
- **Graph Selection:**  
  Users can choose between the Country, City, and Combined graphs.
- **Node Metrics Exploration:**  
  Users can input a node (e.g., Afghanistan) and view metrics such as degree centrality, outgoing edge count, and trap score.
- **Simulation Section:**  
  The dashboard includes an option to simulate turn-based game strategies, showing win rates for various approaches.
- **Community Detection Analysis:**  
  The dashboard provides interactive options to view community graphs (using Louvain or Girvan–Newman), community size distributions, semantic clusters, bridge node analysis, perturbation analysis, and additional creative analyses (e.g., community internal densities and cultural/economic correlations).
- **Letter Frequency Analysis:**  
  Bar charts of starting and ending letter frequencies allow users to see which letters are rare, supporting the development of the letter_gap strategy.
- **Longest Path Visualization:**  
  A DFS-based approach (with a depth cutoff) is used to compute and display the longest path from a selected node, highlighting potential “safe” moves.

---


# 3. Qualitative Insights and Strategic Hypotheses

### Q1: Do the computed strategic metrics (e.g., degree, betweenness, trap score) reveal real strategic nodes?  
- **Observation:**  
  Nodes such as **AFGHANISTAN**, **AZERBAIJAN**, and **ARGENTINA** consistently appear as top nodes in degree centrality, betweenness, and PageRank analyses.  
- **Hypothesis:**  
  These nodes are likely “strategic” because they have a high number of connections (high degree) and serve as bridges (high betweenness), making them key junctions in the network. However, having many connections might also offer the opponent a lot of options.  
- **Discussion:**  
  A high trap score (which is based on low out‑degree of neighbors) might counterbalance this by indicating that although a node has many edges, its neighbors themselves have fewer options, potentially trapping an opponent.  
- **Insight:**  
  In the Atlas game, starting with such nodes could be advantageous—but only if the subsequent moves limit the opponent’s responses. The simulation results (with letter_gap strategy winning 96% of games in one experiment) support the idea that carefully selecting moves based on rare ending letters can be highly effective.

### Q2: Why does the simulation sometimes show a higher win rate for the strategic approach compared to the random strategy, and sometimes the opposite?  
- **Observation:**  
  In one simulation, the Letter Gap Strategy wins 96% of games, while in another simulation using alternative approaches (min_out, composite, lookahead) the win rates vary (e.g., Lookahead 25%, Random 35%).
- **Hypothesis:**  
  This discrepancy may be due to differences in the simulation’s assumptions and the network’s inherent structure. For instance, the Atlas rule creates a network where the connectivity pattern is driven by the linguistic features of place names. In some cases, a simple strategy like letter_gap (focusing on rare starting letters) might exploit this pattern very effectively, while more “sophisticated” strategies (like pure out‑degree maximization) might inadvertently give the opponent too many options.
- **Discussion:**  
  The simulation outcomes indicate that the choice of strategy must account for both the number of moves available from a node and the quality (i.e., restrictiveness) of those moves. The disconnect between theoretical metrics (like high degree centrality) and game outcomes underlines the need for a more nuanced strategy.

### Q3: How does the network structure (countries vs. cities vs. combined graph) affect strategic outcomes?  
- **Observation:**  
  The country graph, with only 249 nodes, appears to have clear strategic nodes (e.g., Afghanistan). However, when cities are added (combined graph), the influence of these strategic nodes is diluted.
- **Hypothesis:**  
  A smaller, more homogeneous network (the country graph) may lead to more predictable and exploitable patterns, while the combined graph introduces noise from the more diverse city names, thereby reducing the clarity of strategic hubs.
- **Discussion:**  
  This suggests that strategies effective on the country graph might not directly transfer to the combined graph. Thus, players might need to adjust their approach depending on the scope of the dataset.
  
### Q4: What do survival path length simulations tell us about node “safety”?  
- **Observation:**  
  Survival path lengths (e.g., around 19 moves for top strategic nodes) indicate how long a player might continue making moves starting from a given node.
- **Hypothesis:**  
  Nodes with longer survival path lengths are “safer” because they have more valid moves before reaching a dead end. However, if a node has a high degree centrality, it may offer many choices—but not necessarily a long path if its neighbors are themselves highly connected.
- **Discussion:**  
  By comparing survival path lengths across different nodes, one could identify “hubs” that allow prolonged play. These hubs might be preferred starting points for strategic players.

### Q5: What improvements can be made based on simulation results?  
- **Observation:**  
  The simulation shows that the letter_gap strategy (which leverages the rarity of starting letters) can dramatically outperform a random strategy.  
- **Hypothesis:**  
  This implies that the Atlas game’s structure is sensitive to linguistic patterns. A refined strategy might combine multiple metrics (degree, trap score, survival length) into a composite heuristic.
- **Discussion:**  
  Future work could involve developing a multi‑criteria decision rule (or even a machine learning model) that predicts the “best” move given a node’s metrics, rather than relying on a single criterion.

---

# 4. Objective Assessment of Community Detection Quality

- **Degree Centrality:**  
  High degree nodes (e.g., Afghanistan, Azerbaijan, Argentina) are consistently identified as strategic. This is a robust metric for initial move selection, but it does not guarantee that moves from these nodes will lead to long survival paths.

- **Betweenness Centrality:**  
  Nodes with high betweenness indicate bridging positions in the network. Such nodes can potentially control game flow. However, high betweenness can also mean that many alternative moves are available, which might benefit an opponent if not exploited carefully.

- **Trap Score:**  
  The custom trap score provides a measure of how “trapping” a node is (i.e., if its neighbors have few outgoing moves). This metric adds nuance to the analysis and could be key in developing an advanced strategy.

- **Graph Density and Clustering Coefficient:**  
  Density measures the ratio of actual edges to possible edges, while the clustering coefficient indicates the degree to which nodes tend to cluster together. These metrics help assess how “tight” a community is.

- **Survival Path Length via Monte Carlo Simulation:**  
  Monte Carlo simulations yield an average survival path length of about 19 moves for top strategic nodes. This metric serves as a proxy for the “safety” of a move, though it may be sensitive to simulation parameters.

- **Simulation Outcomes:**  
  The simulation results (e.g., 96% win rate for the letter_gap strategy in one experiment) suggest that the Atlas game network is highly exploitable when players choose moves based on rare starting letters. This is a promising insight, but the variability in win rates under different strategy definitions indicates that the model is sensitive and may require further refinement.

- **Trade-Offs and Adaptability:**  
  The analysis reveals a potential disconnect: while high out‑degree nodes are theoretically attractive, they can also offer the opponent many options. This underscores the need for a composite strategy that balances multiple factors.

---
# 5. References

- **NetworkX Documentation:** [https://networkx.github.io/documentation/stable/](https://networkx.github.io/documentation/stable/)
- **PyVis Documentation:** [https://pyvis.readthedocs.io/](https://pyvis.readthedocs.io/)
- **Plotly 3D Visualizations:** [https://plotly.com/python/3d-scatter-plots/](https://plotly.com/python/3d-scatter-plots/)
- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **Additional Resources:**  
  - For community detection algorithms and modularity concepts, see relevant academic literature and tutorials.  
  - For Atlas game strategy insights, refer to external blogs such as [How to Win in Atlas](https://imshrinidhi.blogspot.com/2015/10/how-to-win-in-atlas-awesome-tips.html).


---

# **Part 2: Community Detection**

## **1. Key Observations from Outputs**
- **Louvain Method**:
  - **6 communities** with **modularity 0.4633**, indicating strong community structure.
  - Communities vary in size (60, 81, 42, 11, 54, 1 nodes). The smallest community (1 node: likely an outlier) suggests noise.
  - **Continental alignment**: Communities partially group countries by geography (e.g., Community 0 has 14 African countries, Community 5 includes European/North American nations). This suggests that the communities might indeed reflect real‑world groupings
  
- **Girvan-Newman Method**:
  - **8 communities** with **modularity 0.0026**, implying weak partitioning.
  - One giant community (239 nodes) dominates, with others being singletons. This implies that the Girvan–Newman partitioning (at the chosen iteration limit) does not produce a meaningful division of the network. It may be that the algorithm—given the heuristic iteration limit—is not able to separate the network into communities that are denser internally and sparser between them. This “lopsided” partition likely indicates that the method did not find an optimal stopping point under the imposed iteration limit.

## **2. Questions & Hypotheses**
1. **Do communities align with human-interpretable concepts?**
   - **Yes, but partially.** Louvain communities loosely correlate with continents (e.g., African nations cluster together). However, overlaps exist (e.g., Community 5 mixes Europe and North America), possibly due to colonial ties or linguistic similarities.
   - **No semantic alignment**: Linguistic clusters (e.g., "-stan" or "-ia" countries) are scattered across communities. Algorithms prioritize connectivity over semantics.

2. **Can community structure inform game strategy?**
   - **Hypothesis**: A player could choose a starting country from a community with high internal density and low external connectivity, which might trap an opponent into a limited set of moves. For instance, communities with many internally connected nodes (as indicated by a high internal density) could serve as strategic hubs.
   - **Yes.** 
     - **Dense communities**: Starting in high-density groups (e.g., Louvain Community 5, density 0.59) allows longer internal chains before needing external moves.
     - **Sparse communities**: Forcing opponents into low-density areas (e.g., singleton communities) limits their options.
   - **Bridge nodes**: Critical nodes like `YEMEN` (high betweenness) connect disparate regions. Controlling these lets players steer the game into dead-ends.

3. **Why does Girvan-Newman underperform?**
   - The algorithm removes edges iteratively, splitting the graph into trivial components (e.g., singletons). This is unsuitable for the Atlas graph, where most nodes are interconnected.

4. **Are communities robust to perturbations?**
   - **Yes.** Removing 5% of edges increased Louvain modularity slightly (0.4708), indicating stable community definitions. The nearly identical (or slightly higher) modularity after removing 50% of edges suggests that the overall community structure is robust. This robustness implies that the detected communities are not fragile artifacts of minor connectivity patterns but are stable divisions within the network.

### **3. Modularity Assessment**
- **Louvain**:
  - Modularity **0.4633** (values >0.3 indicate strong structure). This confirms meaningful partitioning.
  - Balanced community sizes (except one outlier) suggest the algorithm avoids over-splitting.
- **Girvan-Newman**:
  - Modularity **0.0026** (near 0 implies random partitioning). The method is ineffective here due to the graph’s interconnected nature.

### **4. Strategic Recommendations**
- **Early Game**: Start in high-density communities (e.g., Louvain Community 5) to maximize move options.
- **Mid Game**: Transition via bridge nodes (e.g., `YEMEN`, `OMAN`) to push opponents into sparse regions.
- **Late Game**: Force opponents into singleton communities (e.g., `YEMEN`) to end the game.

### **5. Semantic Clusters**
- Do the semantic clusters (e.g., '-stan', '-land', '-ia') correlate with the algorithmically detected communities?
- **Hypothesis**:
If there is overlap between semantic clusters and algorithmic communities, that would suggest that the naming patterns (which drive the Atlas game rule) have an external linguistic or cultural basis.
- **Observation**:
With 7 '-stan', 11 '-land', and 34 '-ia' countries detected, comparing these lists with the communities from Louvain may reveal if, for instance, most '-stan' countries fall into one community. This would support the idea that the Atlas rule (based on letter patterns) taps into real linguistic groupings.

### **6. Community-Centric Metrics**
- What do the community-centric metrics (internal densities) tell us about the potential strategic zones?
- **Hypothesis:**
Communities with higher internal densities indicate that the countries (or cities) within are tightly connected by the Atlas rule. These might represent “safe zones” or hubs where a player could have many move options, or conversely, force an opponent into a confined region. 
- While using cities_df may capture urban trends rather than national-level data, it still offers valuable insights into how naming patterns correlate with urban concentration.

#### **7. Limitations & Improvements**
- **Data Gaps**: Add GDP/language metadata to test economic/cultural correlations.
- **Algorithm Choice**: Compare with Infomap or Leiden for better resolution.
- **Hybrid Strategies**: Combine community density and bridge-node targeting.

---

![Community Visualization](outputs/graphs/community_comparison.png)  
*Louvain (left) vs. Girvan-Newman (right). Louvain’s communities are cohesive; Girvan-Newman’s are fragmented.*

# Bonus: Link Prediction

For the bonus task, I extended my Atlas game project by exploring whether a neural network can predict if an edge exists between two countries based solely on the network’s structure. In this task, I implemented two approaches: one using Node2vec embeddings and another using a Graph Neural Network (GNN) built with PyTorch Geometric. Both methods were trained in an unsupervised manner since our dataset does not have explicit labels for edge existence.

## Methodology

### Data Preparation and Edge Masking
- **Graph Construction:**  
  I used the original Atlas graph (constructed using the rule that an edge exists from A to B if the last letter of A equals the first letter of B) as the basis for this task.
- **Edge Masking:**  
  I randomly masked 20% of the edges in the graph to simulate missing links. This provided a set of positive examples (the masked edges) and an equal number of negative examples (node pairs that were not originally connected). This masking is crucial for training and evaluating the link prediction models.

### Node2vec-Based Link Prediction
- **Embedding Generation:**  
  I applied the Node2vec algorithm on the masked graph to generate 64‑dimensional embeddings for each node. The algorithm uses random walks and a skip‑gram model to capture the network’s connectivity patterns.
- **Similarity and Evaluation:**  
  I computed cosine similarity between the embeddings of two nodes as a measure of link likelihood. The performance was evaluated using the ROC AUC (Area Under the Receiver Operating Characteristic Curve), which measures how well the model distinguishes between positive (existing) and negative (non‑existent) edges. A high AUC (close to 1) indicates excellent performance.
- **Performance:**  
  The Node2vec approach achieved a ROC AUC of approximately 0.923, demonstrating its strong ability to predict links in the Atlas network.

### GNN-Based Link Prediction using PyTorch Geometric
- **Node Features:**  
  For the GNN, I constructed simple node features by one-hot encoding the first and last letters of each country’s name. This choice was intuitive because the Atlas rule itself is based on these letters.
- **Model Architecture:**  
  I built a two‑layer Graph Convolutional Network (GCN) where:
  - The first layer transforms the input features into a hidden representation using ReLU activation.
  - The second layer generates a lower‑dimensional embedding for each node.
- **Unsupervised Training Objective:**  
  Since we have no ground truth labels for link existence, I used a margin ranking loss. This loss function encourages the model to assign higher scores to positive edges (those that exist) than to negative edges (randomly sampled non‑edges) by a specified margin.
- **Negative Sampling:**  
  For each positive edge in the training set, I generated an equal number of negative edges. This balanced sampling is critical for effective training.
- **Evaluation:**  
  The model’s performance was measured using ROC AUC on a held‑out test set of edges. The training loop runs for 200 epochs, and every 20 epochs the loss and test AUC are printed. For example, by epoch 200 the Test AUC reached around 0.9574 to 0.9702, indicating that the model learned to accurately differentiate positive and negative links.
- **Key Terms Explained:**
  - **Epoch:** One complete pass through the training data.
  - **Loss:** A measure of error; lower loss means the model’s predictions are closer to the true labels. In my experiments, the loss decreased from around 0.28 to 0.055 over the training period.
  - **AUC (Area Under the ROC Curve):** A metric that ranges from 0 to 1. Values closer to 1 indicate better performance; an AUC of 0.5 suggests random guessing.
  
## Performance and Insights

- **Node2vec Performance:**  
  - The Node2vec approach achieved a ROC AUC of ~0.923, which shows that the embeddings capture the connectivity patterns very well.
  
- **GNN Performance:**  
  - The GNN model’s training showed steady improvements with a Test AUC reaching up to ~0.97 by epoch 200.
  - These strong AUC scores indicate that the model is very effective at predicting whether an edge exists, meaning it can distinguish between pairs of countries that are connected and those that are not.

- **Feature Intuition:**  
  - I chose to use one-hot encoding of the first and last letters as features because the Atlas game rule is based on these letters. This simple yet effective feature construction directly ties the model’s input to the fundamental game mechanics.
  
- **Unsupervised Objective:**  
  - By using margin ranking loss in an unsupervised setting, the model learns to prioritize positive edges over negative ones. This method is essential since we do not have labeled data for link existence.
  
- **Overall Insight:**  
  - The bonus task demonstrates that advanced methods like Node2vec and GNNs can capture the underlying structure of the Atlas network. This capability could be extended to predict future moves in the game or to design strategies that exploit the network’s connectivity.

---

Below is an additional section you can append to your README.md file. This section explains how to run the various codes—especially dashboard.py and the bonus task code—in your project.

---

## How to Run the Codes

### Running the Dashboard (dashboard.py)

1. **Prerequisites:**
   - Make sure you have Python 3 installed.
   - Install the required libraries by running:
     ```bash
     pip install streamlit networkx pyvis pandas jinja2 matplotlib numpy
     ```
   - Confirm that the following files and directories exist in your project structure:
     - `notebooks/dashboard.py`
     - `data/countries.csv` and `data/cities.csv`
     - Pre-saved graph files in `notebooks/` (e.g., `country_graph.graphml`, `city_graph.graphml`, etc.)

2. **Running the Dashboard:**
   - Open a terminal or command prompt.
   - Navigate to the project’s root directory (where the README.md file is located).
   - Run the following command:
     ```bash
     streamlit run notebooks/dashboard.py
     ```
   - Your default web browser should open, displaying the interactive dashboard. In the dashboard, you can:
     - Select different graphs (Country, City, Combined).
     - Explore node metrics by entering a node name.
     - Run simulations to compare turn‑based game strategies.
     - View interactive PyVis visualizations and various community detection analyses.

3. **Expected Behavior:**
   - The dashboard will load the specified graphs, display visualizations (2D, 3D, and community graphs), and allow you to interact with simulation settings.
   - If there are any issues, check the terminal output for error messages and verify that your file paths (e.g., for CSV and GraphML files) are correct.

---

### Running the Bonus Task Code (Link Prediction)

1. **Setup in Google Colab:**
   - Create a new notebook (e.g., `bonus_link_prediction.ipynb`).
   - In the first cell, paste the installation commands:
     ```python
     # Install required packages for Node2vec and PyTorch Geometric.
     !pip install node2vec
     !pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
     !pip install torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
     !pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-$(python -c "import torch; print(torch.__version__)").html
     !pip install git+https://github.com/pyg-team/pytorch_geometric.git
     ```
   - Execute the cell and ensure all packages install successfully.

2. **Expected Output:**
   - For the Node2vec part, you should see:
     ```
     Computing transition probabilities: 100%
     249/249 [00:02<00:00, 289.67it/s]
     Node2vec Link Prediction ROC AUC: 0.92296
     ```
   - For the GNN part, you should see printed output every 20 epochs, for example:
     ```
     Seed: 42, Negative edges in training set: 5266
     Training GNN-based Link Prediction Model:
     Epoch: 020, Loss: 0.2824, Test AUC: 0.8747
     Epoch: 040, Loss: 0.1515, Test AUC: 0.9163
     ...
     Epoch: 200, Loss: 0.0722, Test AUC: 0.9574
     ```
     These outputs indicate that the model is learning effectively (with increasing AUC and decreasing loss).
     