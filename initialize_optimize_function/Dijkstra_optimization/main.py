import heapq

def optimized_dijkstra(start_node, graph):
    queue = [(0, start_node)]
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    # Sort and return
    return dict(sorted(distances.items(), key=lambda x: x[1]))

# Example graph
graph_dijkstra = {
    'B': {'A': 5, 'D': 35, 'E': 10},
    'A': {'B': 5, 'C': 10},
    'C': {'A': 10, 'D': 20, 'E': 30},
    'D': {'B': 35, 'C': 20, 'E': 10},
    'E': {'C': 30, 'D': 10, 'B': 10}
}

result_dijkstra = optimized_dijkstra('A', graph_dijkstra)
print("Optimized Dijkstra (Sorted):", result_dijkstra)
