# define dijkstra
def dijkstra(start_node,graph):
    # we define visited and unvisited nodes #
    unvisited = {}
    visited = {}

    for node in graph:
        # initially node distance get infinity
        unvisited[node] = float('inf')
    #print(unvisited) # initially all the nodes distance are infinity

    # Defined the current node as A and Current Distance as 0.
    current = start_node
    currentDistance = 0
    unvisited[current] = currentDistance
    #print(unvisited) #print current unvisited nodes and distances

    while True:
        for neighbor,distance in graph[current].items():
            #print(neighbor,distance) #print B and C nodes neighbor and distances

            # all nodes are cheked skip to next itterration
            if neighbor not in unvisited:
                continue

            #find the new distances(current_Distance + neghbour Distance(distance)
            newDistances = currentDistance+ distance

            #find distance is less than the current distance, we update it
            if unvisited[neighbor] is float('inf') or unvisited[neighbor]>newDistances:
                unvisited[neighbor] = newDistances

        #if we chnged the distance we need to chandged according to unvisited liabrary to visited
        visited[current]= currentDistance
        del unvisited[current]

        #print visited and unvisited
        #print(visited)
        #print(unvisited)

        # if there not any unvisited node we break the loop
        if not unvisited:
            break

        unvisited_items = [node for node in unvisited.items()]
        print(unvisited_items)
        sorted_unvisited_items = sorted(unvisited_items, key=lambda x:x[1])
        #print(sorted_unvisited_items)
        current, currentDistance = sorted_unvisited_items[0]
        print(current,currentDistance)


    return visited
# Definde dictionary and include initial node as unvisited and values is infinity


# We start Node A
start_node='A'

#Define the grapgh
graph={
    'B':{'A':5,'D':35,'E':10},
    'A':{'B':5,'C':10},
    'C':{'A':10,'D':20,'E':30},
    'D':{'B':35,'C':20,'E':10},
    'E':{'C':30,'D':10,'B':10},

}

result=dijkstra(start_node,graph)
print(result)


