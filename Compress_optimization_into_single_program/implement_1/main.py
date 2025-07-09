import heapq
import matplotlib.pyplot as plt
import networkx as nx
import time
import math
from typing import Dict, List, Tuple, Union


class RoutingAlgorithms:
    """
    A class implementing various routing algorithms with visualization and comparison capabilities.

    Attributes:
        steps_dijkstra (list): Stores steps taken during Dijkstra's algorithm execution
        steps_bellman (list): Stores steps taken during Bellman-Ford algorithm execution
    """

    def __init__(self):
        """Initialize the RoutingAlgorithms class with empty step trackers."""
        self.steps_dijkstra = []
        self.steps_bellman = []

    def _validate_graph(self, graph: Dict) -> None:
        """
        Validate the input graph structure.

        Args:
            graph: The graph to validate

        Raises:
            ValueError: If graph is invalid
        """
        if not isinstance(graph, dict):
            raise ValueError("Graph must be a dictionary")

        for node, neighbors in graph.items():
            if not isinstance(neighbors, dict):
                raise ValueError(f"Neighbors of node {node} must be a dictionary")
            for neighbor, weight in neighbors.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight between {node} and {neighbor} must be numeric")

    def dijkstra(self, start_node: str, graph: Dict) -> Dict[str, float]:
        """
        Optimized Dijkstra's algorithm using priority queue.

        Args:
            start_node: Starting node for path calculation
            graph: Graph represented as adjacency list

        Returns:
            Dictionary of shortest distances from start_node to all other nodes

        Raises:
            ValueError: If start_node not in graph or graph is invalid
        """
        self._validate_graph(graph)
        if start_node not in graph:
            raise ValueError(f"Start node {start_node} not found in graph")

        self.steps_dijkstra = []
        queue = [(0, start_node)]
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        visited = set()

        while queue:
            try:
                current_distance, current_node = heapq.heappop(queue)
            except IndexError:
                break

            if current_node in visited:
                continue

            visited.add(current_node)
            self.steps_dijkstra.append((current_node, current_distance, dict(distances)))

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))

        return dict(sorted(distances.items(), key=lambda x: x[1]))

    def dijkstra_initial(self, start_node: str, graph: Dict) -> Dict[str, float]:
        """
        Initial implementation of Dijkstra's algorithm without priority queue.

        Args:
            start_node: Starting node for path calculation
            graph: Graph represented as adjacency list

        Returns:
            Dictionary of shortest distances from start_node to all other nodes
        """
        self._validate_graph(graph)
        if start_node not in graph:
            raise ValueError(f"Start node {start_node} not found in graph")

        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        visited = set()

        while len(visited) < len(graph):
            current_node = None
            for node in graph:
                if node not in visited and (current_node is None or distances[node] < distances[current_node]):
                    current_node = node

            if current_node is None:
                break

            visited.add(current_node)

            for neighbor, weight in graph[current_node].items():
                if distances[current_node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[current_node] + weight

        return distances

    def bellman_ford(self, start_node: str, graph: Dict) -> Union[Dict[str, float], str]:
        """
        Optimized Bellman-Ford algorithm with early termination.

        Args:
            start_node: Starting node for path calculation
            graph: Graph represented as adjacency list

        Returns:
            Dictionary of shortest distances or error message if negative cycle detected
        """
        self._validate_graph(graph)
        if start_node not in graph:
            raise ValueError(f"Start node {start_node} not found in graph")

        self.steps_bellman = []
        distance = {node: float('inf') for node in graph}
        distance[start_node] = 0
        edges = [(u, v, w) for u in graph for v, w in graph[u].items()]

        for i in range(len(graph) - 1):
            updated = False
            for u, v, w in edges:
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
                    updated = True
                    self.steps_bellman.append((i + 1, u, v, distance[v], dict(distance)))
            if not updated:
                break

        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                return "Negative weight cycle detected"

        return dict(sorted(distance.items(), key=lambda x: x[1]))

    def bellman_ford_initial(self, start_node: str, graph: Dict) -> Union[Dict[str, float], str]:
        """
        Initial implementation of Bellman-Ford algorithm without optimizations.

        Args:
            start_node: Starting node for path calculation
            graph: Graph represented as adjacency list

        Returns:
            Dictionary of shortest distances or error message if negative cycle detected
        """
        self._validate_graph(graph)
        if start_node not in graph:
            raise ValueError(f"Start node {start_node} not found in graph")

        distance = {node: float('inf') for node in graph}
        distance[start_node] = 0
        edges = [(u, v, w) for u in graph for v, w in graph[u].items()]

        for _ in range(len(graph) - 1):
            for u, v, w in edges:
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w

        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                return "Negative weight cycle detected"

        return distance

    def visualize_graph(self, graph: Dict, title: str = "Graph") -> None:
        """
        Visualize the graph using NetworkX and Matplotlib.

        Args:
            graph: Graph to visualize
            title: Title for the visualization
        """
        try:
            self._validate_graph(graph)
            G = nx.DiGraph()

            for node in graph:
                for neighbor, weight in graph[node].items():
                    G.add_edge(node, neighbor, weight=weight)

            pos = nx.spring_layout(G)
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_color='skyblue',
                    node_size=1200, font_size=12, arrows=True)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.title(title)
            plt.show()

        except Exception as e:
            print(f"Error visualizing graph: {e}")

    def compare_performance(self, graph: Dict, start_node: str) -> None:
        """
        Compare performance of all algorithm implementations on increasing graph sizes.

        Args:
            graph: Complete graph to test
            start_node: Starting node for algorithms
        """
        try:
            self._validate_graph(graph)
            if start_node not in graph:
                raise ValueError(f"Start node {start_node} not found in graph")

            sizes = range(2, len(graph) + 1)
            results = {
                "Dijkstra_Initial": [],
                "Dijkstra_Optimized": [],
                "Bellman_Initial": [],
                "Bellman_Optimized": [],
                "Nodes": [],
                "Edges": []
            }

            for size in sizes:
                nodes_subset = list(graph.keys())[:size]
                subgraph = {
                    node: {nbr: w for nbr, w in graph[node].items() if nbr in nodes_subset}
                    for node in nodes_subset
                }

                V = len(subgraph)
                E = sum(len(neighbors) for neighbors in subgraph.values())
                results["Nodes"].append(V)
                results["Edges"].append(E)

                # Time each algorithm
                algorithms = [
                    ("Dijkstra_Initial", self.dijkstra_initial),
                    ("Dijkstra_Optimized", self.dijkstra),
                    ("Bellman_Initial", self.bellman_ford_initial),
                    ("Bellman_Optimized", self.bellman_ford)
                ]

                for name, func in algorithms:
                    start = time.perf_counter()
                    func(start_node, subgraph)
                    results[name].append(time.perf_counter() - start)

            # Plot results
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            colors = ['blue', 'green', 'red', 'purple']

            for i, (name, color) in enumerate(zip([
                "Dijkstra_Initial",
                "Dijkstra_Optimized",
                "Bellman_Initial",
                "Bellman_Optimized"
            ], colors)):
                ax = axs[i // 2, i % 2]
                ax.plot(results["Nodes"], results[name], marker='o', color=color, label=name)
                ax.set_title(name.replace('_', ' '))
                ax.set_xlabel('Number of Nodes')
                ax.set_ylabel('Execution Time (s)')
                ax.grid(True)
                ax.legend()

            plt.suptitle("Algorithm Performance Comparison", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"Error in performance comparison: {e}")

    def plot_complexity_trend(self, max_nodes: int = 10000) -> None:
        """
        Plot theoretical time complexity trends for the algorithms.

        Args:
            max_nodes: Maximum number of nodes to plot (default: 10,000)
        """
        try:
            # Generate node counts on a logarithmic scale
            node_counts = [10, 100, 1000, 10000][
                          :next((i for i, x in enumerate([10, 100, 1000, 10000]) if x > max_nodes), 4)]

            # Calculate theoretical operations
            results = {
                "Dijkstra_Initial": [],
                "Dijkstra_Optimized": [],
                "Bellman_Initial": [],
                "Bellman_Optimized": [],
                "Nodes": node_counts
            }

            for V in node_counts:
                E = V * 2  # Assume sparse graph
                results["Dijkstra_Initial"].append(V ** 2)
                results["Dijkstra_Optimized"].append((V + E) * math.log2(V) if V > 1 else 0)
                results["Bellman_Initial"].append(V * E)
                results["Bellman_Optimized"].append(V * E)  # Same complexity but may terminate early

            # Normalize values for better comparison
            max_val = max(max(results["Dijkstra_Initial"]),
                          max(results["Dijkstra_Optimized"]),
                          max(results["Bellman_Initial"]),
                          max(results["Bellman_Optimized"]))

            for key in results:
                if key != "Nodes":
                    results[key] = [x / max_val for x in results[key]]

            # Plot results
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            colors = ['blue', 'green', 'red', 'purple']

            for i, (name, color) in enumerate(zip([
                "Dijkstra_Initial",
                "Dijkstra_Optimized",
                "Bellman_Initial",
                "Bellman_Optimized"
            ], colors)):
                ax = axs[i // 2, i % 2]
                ax.plot(results["Nodes"], results[name], marker='o', color=color, label=name)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(f"{name.replace('_', ' ')} Complexity")
                ax.set_xlabel('Number of Nodes (log scale)')
                ax.set_ylabel('Normalized Operations (log scale)')
                ax.grid(True, which='both', linestyle='--')
                ax.legend()

            plt.suptitle("Theoretical Time Complexity Trends (Log-Log Scale)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"Error plotting complexity trends: {e}")

    def visualize_algorithm_steps(self, algorithm: str = 'dijkstra') -> None:
        """
        Visualize the steps taken during algorithm execution.

        Args:
            algorithm: Which algorithm steps to visualize ('dijkstra' or 'bellman')
        """
        try:
            if algorithm.lower() == 'dijkstra':
                steps = self.steps_dijkstra
                if not steps:
                    print("No Dijkstra steps recorded. Run the algorithm first.")
                    return

                print("\nDijkstra's Algorithm Steps:")
                print("{:<10} {:<10} {:<30}".format("Node", "Distance", "Current Distances"))
                for node, dist, distances in steps:
                    dist_str = ", ".join(f"{k}:{v:.1f}" for k, v in distances.items())
                    print(f"{node:<10} {dist:<10.1f} {dist_str:<30}")

            elif algorithm.lower() == 'bellman':
                steps = self.steps_bellman
                if not steps:
                    print("No Bellman-Ford steps recorded. Run the algorithm first.")
                    return

                print("\nBellman-Ford Algorithm Steps:")
                print("{:<5} {:<5} {:<5} {:<10} {:<30}".format(
                    "Iter", "From", "To", "New Dist", "Current Distances"))
                for iter, u, v, new_dist, distances in steps:
                    dist_str = ", ".join(f"{k}:{v:.1f}" for k, v in distances.items())
                    print(f"{iter:<5} {u:<5} {v:<5} {new_dist:<10.1f} {dist_str:<30}")

            else:
                print("Invalid algorithm choice. Use 'dijkstra' or 'bellman'")

        except Exception as e:
            print(f"Error visualizing steps: {e}")


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    try:
        router = RoutingAlgorithms()

        # Sample graphs
        graph_dijkstra = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }

        graph_bellman = {
            'A': {'B': -1, 'C': 4},
            'B': {'C': 3, 'D': 2, 'E': 2},
            'C': {},
            'D': {'B': 1, 'C': 5},
            'E': {'D': -3}
        }

        # Visualize graphs
        print("\nVisualizing Dijkstra's graph:")
        router.visualize_graph(graph_dijkstra, "Graph for Dijkstra's Algorithm")

        print("\nVisualizing Bellman-Ford graph:")
        router.visualize_graph(graph_bellman, "Graph for Bellman-Ford Algorithm")

        # Run algorithms and show steps
        print("\nRunning Dijkstra's algorithm from node 'A':")
        dijkstra_result = router.dijkstra('A', graph_dijkstra)
        print("Dijkstra's result:", dijkstra_result)
        router.visualize_algorithm_steps('dijkstra')

        print("\nRunning Bellman-Ford algorithm from node 'A':")
        bellman_result = router.bellman_ford('A', graph_bellman)
        print("Bellman-Ford result:", bellman_result)
        router.visualize_algorithm_steps('bellman')

        # Compare performance
        print("\nComparing algorithm performance:")
        router.compare_performance(graph_dijkstra, 'A')

        # Show complexity trends
        print("\nPlotting theoretical complexity trends:")
        router.plot_complexity_trend()

    except Exception as e:
        print(f"An error occurred in example usage: {e}")