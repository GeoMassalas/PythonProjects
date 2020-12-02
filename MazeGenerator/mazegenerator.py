import numpy as np
import cv2
from collections import deque
from random import choice

class Maze:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = [['rlud' for j in range(self.width)] for i in range(self.height)]

        self.cell_side = 32
        self.color = 0
        self.background = 255
        self.offset = 2

    def break_wall(self, row, column, direction):
        self.grid[row][column] = self.grid[row][column].replace(direction, "")

    def connect_cells(self, cell_a, cell_b):
        if cell_a != cell_b:
            if cell_a[0] == cell_b[0]:
                if cell_a[1] - cell_b[1] == 1:
                    self.break_wall(cell_a[0], cell_a[1], 'l')
                    self.break_wall(cell_b[0], cell_b[1], 'r')
                elif cell_b[1] - cell_a[1] == 1:
                    self.break_wall(cell_a[0], cell_a[1], 'r')
                    self.break_wall(cell_b[0], cell_b[1], 'l')
            if cell_a[1] == cell_b[1]:
                if cell_a[0] - cell_b[0] == 1:
                    self.break_wall(cell_a[0], cell_a[1], 'u')
                    self.break_wall(cell_b[0], cell_b[1], 'd')
                elif cell_b[0] - cell_a[0] == 1:
                    self.break_wall(cell_a[0], cell_a[1], 'd')
                    self.break_wall(cell_b[0], cell_b[1], 'u')

    def print_maze(self):
        print("")
        for r in range(self.height):
            temp = "  "
            for c in range(self.width):
                temp += self.grid[r][c].ljust(6)
            print(temp)
            print("")

    def paint_cell(self, img, r, c):

        corners = {"nw": (2+c*32, 2+r*32), "sw": (2+c*32, 2+(r+1)*32), "se":(2+(c+1)*32, 2+(r+1)*32), "ne":(2+(c+1)*32, 2+r*32)}
 
        for char in self.grid[r][c]:
            if char == 'r':
                img = cv2.line(img, corners["ne"], corners["se"], self.color, self.offset)
            if char == 'l':           
                img = cv2.line(img, corners["nw"], corners["sw"], self.color, self.offset)
            if char == 'u':
                img = cv2.line(img, corners["nw"], corners["ne"], self.color, self.offset)
            if char == 'd':
                img = cv2.line(img, corners["sw"], corners["se"], self.color, self.offset)

        return img

    def print_canvas(self, name):
        max_width = self.cell_side*self.width + self.offset*2
        max_height = self.cell_side*self.height + self.offset*2
        img = np.zeros((max_height + 1, max_width + 1)) + self.background
        img = cv2.line(img, (0,0), (max_width,0), self.color, self.offset)
        img = cv2.line(img, (0,0), (0,max_height), self.color, self.offset)
        img = cv2.line(img, (max_width, max_height), (max_width,0), self.color, self.offset)
        img = cv2.line(img, (max_width, max_height), (0,max_height), self.color, self.offset)
        for r in range(0, self.height):
            for c in range(0, self.width):
                self.paint_cell(img, r, c)
        cv2.imwrite(name + '.png', img)

class Cell:

    def __init__(self, position, parent=None):
        self.row = position[0]
        self.column = position[1]
        self.parent = parent

    def get_position(self):
        return (self.row, self.column)

    def get_neighbors(self):
        positions = []
        if self.row + 1 < MAX_ROW:
            positions.append((self.row+1,self.column))
        if self.row - 1 >= 0:
            positions.append((self.row-1,self.column))
        if self.column + 1 < MAX_COLUMN:
            positions.append((self.row,self.column+1))
        if self.column - 1 >= 0:
            positions.append((self.row,self.column-1))
        return positions

def dfs(maze):
    parent_cell = Cell((0,0))
    init_cell = Cell((0,0), parent_cell)
    explored = set()
    stack = deque()
    stack.append(init_cell)
    while(len(stack) > 0):
        current_cell = stack.pop()
        if current_cell.get_position() not in explored:
            maze.connect_cells(current_cell.get_position(), current_cell.parent.get_position())
            #maze.print_canvas('test')
            #_ = input("Press any key to continue!")
            next_positions = current_cell.get_neighbors()
            while(len(next_positions) > 0):
                pos = choice(next_positions)
                if pos not in explored:
                    new_cell = Cell(pos, current_cell)
                    stack.append(new_cell)
                next_positions.remove(pos)
            explored.add(current_cell.get_position())
    return maze

def generate_maze(rows, columns, method):
    if method == "dfs":
        init_maze = Maze(rows, columns)
        maze = dfs(init_maze)
    return maze

if __name__ == "__main__":
    MAX_ROW = 36*4
    MAX_COLUMN = 66*4
    maze = generate_maze(MAX_ROW, MAX_COLUMN, "dfs")
    maze.print_canvas('test')
    maze.print_maze()
    print("Done")
