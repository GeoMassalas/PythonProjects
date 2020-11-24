from collections import deque
import argparse
import heapq
import time
import resource

# Target
target = [1,2,3,4,5,6,7,8,0]
# Number to index dictionary
index_to_number = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

class State:

    def __init__(self, state, depth=1, parent=None):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.est_cost = self.get_manhatan_distance() + depth

    def __lt__(self, other):
        return self.est_cost < other.est_cost

    def get_state(self):
        return self.state

    def get_depth(self):
        return self.depth

    def get_parent(self):
        return self.parent

    def get_next_moves(self):
        zero = self.state.index(0)
        new_states = []
        if zero > 2:
            new_state = self.state[:]
            new_state[zero-3], new_state[zero] = new_state[zero], new_state[zero-3]
            if not self.is_parent_state(new_state):
                new_states.append(new_state)

        if zero < 6:
            new_state = self.state[:]
            new_state[zero+3], new_state[zero] = new_state[zero], new_state[zero+3]
            if not self.is_parent_state(new_state):
                new_states.append(new_state)

        if zero % 3 != 0:
            new_state = self.state[:]
            new_state[zero-1], new_state[zero] = new_state[zero], new_state[zero-1]
            if not self.is_parent_state(new_state):
                new_states.append(new_state)

        if zero % 3 != 2:
            new_state = self.state[:]
            new_state[zero+1], new_state[zero] = new_state[zero], new_state[zero+1]
            if not self.is_parent_state(new_state):
                new_states.append(new_state)

        return new_states

    def is_parent_state(self, state):
        if self.parent:
            return self.parent.get_state() == state 
        else:
            return False

    def get_manhatan_distance(self):
        manh_sum = 0
        for i in range(9):
            if self.state[i] != 0:
                index_dif = abs(i - index_to_number[self.state[i]])
                manh_sum += index_dif % 3 + index_dif // 3
        return manh_sum

def get_arguments():
    """ This function parses and returns the arguments provided. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--puzzle",
                        dest="puzzle",
                        help="Puzzle should be nine unique numbers(0-8) seperated only by commas."
                        )
    args = parser.parse_args()
    if not args.puzzle:
        parser.error("Please add a puzzle with -p.")
    if "".join(sorted(args.puzzle.split(","))) == "012345678":
        return [int(i) for i in args.puzzle.split(",")]
    else:
        parser.error("Invalid Puzzle. Puzzle should be nine unique numbers(0-8) seperated only by commas.")


def verbose_path(path):
    last = path[0]
    moves = []
    for i in range(0, len(path)):
        l = last.index(0)
        n = path[i].index(0)
        if n - l == 1:
            moves.append("RIGHT")
        elif n - l == -1:
            moves.append("LEFT")
        elif n - l == 3:
            moves.append("DOWN")
        elif n - l == -3:
            moves.append("UP")
        last = path[i]
    if len(moves) > 40:
        return moves[:20] + ["... " + str(len(moves)-40) + " more in between moves ..."] + moves[-20:]
    else:
        return moves
        
def bfs(puzzle_state):
    state = State(puzzle_state, 1, None)
    q = deque()
    q.append(state)
    explored = set()
    max_depth = 1
    t = time.time()
    while(len(q) > 0):
        state = q.popleft()        
        if state.get_depth() > max_depth:
            max_depth = state.get_depth()
        if state.get_state() == target:
            elapsed_time = time.time() - t
            f = []
            depth = state.get_depth()
            f.append(state.get_state())
            while(state.get_parent() != None):
                state = state.get_parent()
                f.append(state.get_state())
            return f[::-1], len(explored), depth, elapsed_time, max_depth, len(q)
        else:
            states = state.get_next_moves()
            for i in range(len(states)):
                if tuple(states[i]) not in explored:
                    q.append(State(states[i], state.get_depth() + 1, state))
                    explored.add(tuple(states[i]))

def dfs(puzzle_state):
    
    state = State(puzzle_state, 1, None)
    stack = deque()
    stack.append(state)
    explored = set()
    max_depth = 1
    t = time.time()
    while(len(stack) > 0):
        state = stack.pop()
        if state.get_depth() > max_depth:
            max_depth = state.get_depth()
        if state.get_state() == target:
            elapsed_time = time.time() - t
            f = []
            depth = state.get_depth()
            f.append(state.get_state())
            while(state.get_parent() != None):
                state = state.get_parent()
                f.append(state.get_state())
            return f[::-1], len(explored), depth, elapsed_time, max_depth, len(stack)
        else:
            states = state.get_next_moves()
            for st in states[::-1]:
                if tuple(st) not in explored:
                    stack.append(State(st, state.get_depth() + 1, state))
                    explored.add(tuple(st))


def ast(puzzle_state):
    state = State(puzzle_state, 1, None)
    heap = []
    heapq.heapify(heap)
    heapq.heappush(heap, state)
    explored = set()
    max_depth = 1
    t = time.time()
    while(len(heap) > 0):
        state = heapq.heappop(heap)
        if state.get_depth() > max_depth:
            max_depth = state.get_depth()
        if state.get_state() == target:   
            elapsed_time = time.time() - t
            f = []
            depth = state.get_depth()
            f.append(state.get_state())
            while(state.get_parent() != None):
                state = state.get_parent()
                f.append(state.get_state())
            return f[::-1], len(explored), depth, elapsed_time, max_depth, len(heap)
        else:
            states = state.get_next_moves()
            for st in states:
                if tuple(st) not in explored:
                    heapq.heappush(heap, State(st, state.get_depth() +1, state))
                    explored.add(tuple(st))

def print_diagnostics(algorithm, data):
    print("----------------")
    print(algorithm + " statistics")
    print("----------------")
    print("Path")
    print(verbose_path(data[0]))
    print("Max Depth: " + str(data[4]))
    print("Current Depth: " + str(data[2]))
    print("Nodes Explored: " + str(data[1]))
    print("Nodes Opened: " + str(data[5] + data[1])) 
    print("Time elapsed: " + str("{:.3f}".format(data[3])) + " seconds")
    print("---------------")

if __name__ == "__main__":
    init_puzzle = get_arguments()
    print(init_puzzle)
    bfsdata = bfs(init_puzzle)
    print_diagnostics("BFS", bfsdata)
    _ = input("Press any key to proceed with DFS.")
    dfsdata = dfs(init_puzzle)
    print_diagnostics("DFS", dfsdata)
    _ = input("Press any key to proceed with A*.")
    astdata = ast(init_puzzle)
    print_diagnostics("A*", astdata)

