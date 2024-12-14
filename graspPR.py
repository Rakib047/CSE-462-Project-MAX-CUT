import random
import numpy as np
import networkx as nx

# Helper function to calculate the cut value
def calculate_cut_value(graph, partition):
    cut_value = 0
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            cut_value += data.get('weight', 1)
    return cut_value

# Construction phase of GRASP
def construct_initial_solution(graph):
    partition = {node: random.choice([0, 1]) for node in graph.nodes}
    return partition

# Local search phase of GRASP
def local_search(graph, partition):
    improved = True
    while improved:
        improved = False
        for node in graph.nodes:
            current_cut_value = calculate_cut_value(graph, partition)
            partition[node] = 1 - partition[node]  # Flip node's partition
            new_cut_value = calculate_cut_value(graph, partition)
            if new_cut_value > current_cut_value:
                improved = True
            else:
                partition[node] = 1 - partition[node]  # Undo flip if no improvement
    return partition

# Path relinking phase
def path_relinking(graph, solution_a, solution_b):
    best_solution = solution_a.copy()
    best_cut_value = calculate_cut_value(graph, best_solution)

    current_solution = solution_a.copy()
    for node in solution_a:
        if solution_a[node] != solution_b[node]:
            current_solution[node] = solution_b[node]
            current_cut_value = calculate_cut_value(graph, current_solution)
            if current_cut_value > best_cut_value:
                best_solution = current_solution.copy()
                best_cut_value = current_cut_value

    return best_solution

# GRASP with Path Relinking
def grasp_pr(graph, max_iterations, elite_set_size):
    elite_set = []
    best_solution = None
    best_cut_value = -float('inf')

    for iteration in range(max_iterations):
        # GRASP Phase
        initial_solution = construct_initial_solution(graph)
        refined_solution = local_search(graph, initial_solution)

        # Update elite set
        refined_cut_value = calculate_cut_value(graph, refined_solution)
        if len(elite_set) < elite_set_size:
            elite_set.append((refined_solution, refined_cut_value))
        else:
            worst_index = min(range(len(elite_set)), key=lambda i: elite_set[i][1])
            if refined_cut_value > elite_set[worst_index][1]:
                elite_set[worst_index] = (refined_solution, refined_cut_value)

        # Path Relinking Phase
        for elite_solution, _ in elite_set:
            if refined_solution != elite_solution:
                candidate_solution = path_relinking(graph, refined_solution, elite_solution)
                candidate_cut_value = calculate_cut_value(graph, candidate_solution)

                if candidate_cut_value > best_cut_value:
                    best_solution = candidate_solution
                    best_cut_value = candidate_cut_value

    return best_solution, best_cut_value

# Example usage
if __name__ == "__main__":
    # Create a random graph for the MAX CUT problem
    num_nodes = 10
    edge_probability = 0.5
    graph = nx.erdos_renyi_graph(num_nodes, edge_probability)

    # Add random weights to edges
    for u, v in graph.edges:
        graph[u][v]['weight'] = random.randint(1, 10)

    max_iterations = 50
    elite_set_size = 5

    best_solution, best_cut_value = grasp_pr(graph, max_iterations, elite_set_size)

    print("Best Solution:", best_solution)
    print("Best Cut Value:", best_cut_value)
