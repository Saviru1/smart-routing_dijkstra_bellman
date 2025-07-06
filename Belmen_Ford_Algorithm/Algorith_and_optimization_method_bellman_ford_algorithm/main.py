
#define the method bellman_ford
def bellman_ford(start_node,graph):
    #create the dictornary
    distance={}

    for node in graph:
        distance[node] = float('inf')
    #print(distance) # initial all nodes declar to the distance value of infinity


    current_distance=0
    distance[start_node] = current_distance # after initiate the current distance
    #print(distance)

    #graph itteration should be = (no of nodes) - 1
    for i in range(len(graph)-1):
        for node in graph:
            for neighbour in graph[node]:
                if distance[neighbour] > distance[node]+graph[node][neighbour]:
                    distance[neighbour] = distance[node]+graph[node][neighbour]
    for node in graph:
        for neighbour in graph[node]:
            if distance[neighbour] > distance[node] + graph[node][neighbour]:
                return "There is a Negative cycle"


    return distance

# define the graph
start_node = 'A'
graph = {
    'A':{'B':3,'C':5},
    'B':{'C':3},
    'C':{'D':4},
    'D':{'B':-2}
}

result=bellman_ford(start_node,graph)
print(result)