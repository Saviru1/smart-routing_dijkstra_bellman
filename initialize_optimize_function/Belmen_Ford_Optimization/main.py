def optimized_bellman_ford(start_node, graph):
    distance = {node: float('inf') for node in graph}
    distance[start_node] = 0

    edges = [(u, v, w) for u in graph for v, w in graph[u].items()]

    for _ in range(len(graph) - 1):
        updated = False
        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                updated = True
        if not updated:
            break

    for u, v, w in edges:
        if distance[u] + w < distance[v]:
            return "Negative weight cycle detected"

    return dict(sorted(distance.items(), key=lambda x: x[1]))

# Example graph
graph_bellman = {
    'A': {'B': 3, 'C': 5},
    'B': {'C': 3},
    'C': {'D': 4},
    'D': {'B': -2}
}

result_bellman = optimized_bellman_ford('A', graph_bellman)
print("Optimized Bellman-Ford (Sorted):", result_bellman)
