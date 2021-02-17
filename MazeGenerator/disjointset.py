class Node:
    
    def __init__(self, rank, value, parent=None):
        self.rank = rank
        self.parent = parent
        self.value = value

    def set_parent(self, parent):
        self.parent = parent


class DisjointSet:

    def __init__(self, items):
        self.nodes = {}
        for i in items:
            self.nodes[i] = Node(0, i)

    def find(self, node):
        if self.nodes[node].parent is not None:
            return self.find(self.nodes[node].parent.value)
        return self.nodes[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1.value == root2.value:
            return True
        else:
            if root1.rank >= root2.rank:
                root1.rank += 1
                root2.parent = root1
            else:
                root2.rank += 1
                root1.parent = root2
            return False
