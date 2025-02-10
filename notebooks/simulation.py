import random

def get_valid_moves(graph, current, visited):
    """Return valid successors of current that have not been visited."""
    return [n for n in graph.successors(current) if n not in visited]

# Random strategy move: simply chooses a random valid move.
def random_move(graph, current, visited):
    moves = get_valid_moves(graph, current, visited)
    return random.choice(moves) if moves else None

# Letter Gap Strategy move:
# For each valid move, we look at its last letter and choose the move for which the number 
# of nodes starting with that letter (in the full graph) is minimal.
def letter_gap_move(graph, current, visited, start_counts):
    moves = get_valid_moves(graph, current, visited)
    if not moves:
        return None
    # For each candidate move, get its last letter and then fetch the number of nodes starting with that letter.
    best_move = min(moves, key=lambda m: start_counts.get(m[-1], 0))
    return best_move

def compute_start_counts(graph):
    """Precompute a dictionary: letter -> count of nodes that start with that letter."""
    counts = {}
    for node in graph.nodes():
        letter = node[0]
        counts[letter] = counts.get(letter, 0) + 1
    return counts

# Turn-based game simulation between two strategies.
# Each strategy is a function taking (graph, current, visited, start_counts) and returning a move.
def simulate_turn_based_game(graph, strat_func1, strat_func2, start_counts, starting_player=1, verbose=False):
    current = random.choice(list(graph.nodes()))
    visited = {current}
    if verbose:
        print(f"Starting node: {current}")
    turn = starting_player  # 1 or 2
    while True:
        if turn == 1:
            move = strat_func1(graph, current, visited, start_counts)
        else:
            move = strat_func2(graph, current, visited, start_counts)
        if move is None:
            if verbose:
                print(f"Player {turn} cannot move. Winner: Player {3 - turn}")
            return 3 - turn  # The other player wins
        if verbose:
            print(f"Player {turn} chooses: {move}")
        current = move
        visited.add(current)
        turn = 1 if turn == 2 else 2  # switch turns

# Define strategy wrappers to match the required signature.
def random_strategy(graph, current, visited, start_counts):
    return random_move(graph, current, visited)

def letter_gap_strategy(graph, current, visited, start_counts):
    return letter_gap_move(graph, current, visited, start_counts)

# Simulate multiple turn-based games between Random and Letter Gap strategies.
def simulate_turn_based_games(graph, num_games=100, verbose=False):
    start_counts = compute_start_counts(graph)
    wins_random = 0
    wins_letter_gap = 0
    for i in range(num_games):
        # Randomly decide which strategy starts.
        starting_player = random.choice([1, 2])
        if starting_player == 1:
            winner = simulate_turn_based_game(graph, random_strategy, letter_gap_strategy, start_counts, starting_player=1, verbose=False)
            if winner == 1:
                wins_random += 1
            else:
                wins_letter_gap += 1
        else:
            winner = simulate_turn_based_game(graph, letter_gap_strategy, random_strategy, start_counts, starting_player=1, verbose=False)
            if winner == 1:
                wins_letter_gap += 1
            else:
                wins_random += 1
    print(f"Turn-based simulation over {num_games} games:")
    print(f"Random Strategy wins: {wins_random} ({wins_random/num_games*100:.1f}%)")
    print(f"Letter Gap Strategy wins: {wins_letter_gap} ({wins_letter_gap/num_games*100:.1f}%)")
    