import numpy as np
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import argparse
import os
import sys
from tqdm import tqdm
from os.path import isfile, join
from collections import deque
from random import choice, shuffle
import uuid

class Maze:

    def __init__(self, height, width, cell_size=32, color=(0, 0, 0), offset=2, background=255):
        self.height = height
        self.width = width
        self.grid = [['rlud' for j in range(self.width)] for i in range(self.height)]

        self.cell_size = cell_size
        self.color = color
        self.background = background
        self.offset = offset

    def get_number_of_cells(self):
        return self.height * self.width

    def get_rows(self):
        return self.height

    def get_cols(self):
        return self.width

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

        corners = {"nw": (2+c*self.cell_size, 2+r*self.cell_size), 
                "sw": (2+c*self.cell_size, 2+(r+1)*self.cell_size), 
                "se":(2+(c+1)*self.cell_size, 2+(r+1)*self.cell_size), 
                "ne":(2+(c+1)*self.cell_size, 2+r*self.cell_size)}
 
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

    def generate_image_state(self):
        max_width = self.cell_size*self.width + self.offset*2
        max_height = self.cell_size*self.height + self.offset*2
        img = np.zeros([max_height + 1, max_width + 1, 3], dtype=np.uint8 )
        img.fill(self.background)
        img = cv2.line(img, (0,0), (max_width,0), self.color, self.offset)
        img = cv2.line(img, (0,0), (0,max_height), self.color, self.offset)
        img = cv2.line(img, (max_width, max_height), (max_width,0), self.color, self.offset)
        img = cv2.line(img, (max_width, max_height), (0,max_height), self.color, self.offset)
        return img

    def get_canvas(self):
        img = self.generate_image_state()
        for r in range(0, self.height):
            for c in range(0, self.width):
                img = self.paint_cell(img, r, c)
        return img

    def print_canvas(self, name):
        img = self.get_canvas()
        resolution = (1920, 1080)
        resized_img = cv2.resize(img, resolution)
        cv2.imwrite(name + '.png', resized_img)


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


def get_arguments():
    parser = argparse.ArgumentParser(description='Maze Generator script for creating a video of a maze generation algorithm.')
    parser.add_argument('-s','--size', 
            dest='size', type=int, required=True,
            help='Size of the maze in 16:9 incerements. For Example 2 -> 32:18. Range: 1-10')
    parser.add_argument('-b', '--batch-size',
            dest='batch_size', type=int,
            help='Batch size for the video generation. Default Batch: 1000.')
    parser.add_argument('-a', '--algorithm', 
            dest='algo', type=str,
            help="Algorithm to use in maze generation. Default option - dfs")

    args = parser.parse_args()
    if (args.size < 1) | (args.size > 10):
        parser.error("Size needs to be between 1 and 10.")
    else:
        size = 16*9*args.size*args.size
    if not args.batch_size:
        args.batch_size = 1000
    else:
        if (args.batch_size < 100) | (args.batch_size > size):
            parser.error("Batch size needs to be between 100 and maze size(Current Size: {}).".format(size))
    #TODO: algo selection
    args.algo = 'kruskals'
    return args.size, args.batch_size, args.algo

def get_walls(maze):
    walls = []
    for r in range(0,maze.get_rows()):
        for c in range(0,maze.get_cols()):
            if r < maze.get_rows() - 1:
                walls.append(((r, c), (r+1, c)))
            if c < maze.get_cols() - 1:
                walls.append(((r, c), (r, c+1)))
    return walls

def kruskals(maze, image_folder):
    walls = get_walls(maze)
    f = list(range(len(walls)))
    shuffle(f)
    for i in f:
        print(walls[i])
    sys.exit()


def dfs(maze, image_folder):
    parent_cell = Cell((0,0))
    init_cell = Cell((0,0), parent_cell)
    explored = set()
    stack = deque()
    stack.append(init_cell)
    count = 0
    print("Generating Maze Images...")
    pbar = tqdm(total=maze.get_number_of_cells())
    while(len(stack) > 0):
        current_cell = stack.pop()
        if current_cell.get_position() not in explored:
            maze.connect_cells(current_cell.get_position(), current_cell.parent.get_position())
            maze.print_canvas('{}image{}'.format(image_folder, str(count).zfill(len(str(maze.get_number_of_cells())))))
            count += 1
            pbar.update(1)
            next_positions = current_cell.get_neighbors()
            while(len(next_positions) > 0):
                pos = choice(next_positions)
                if pos not in explored:
                    new_cell = Cell(pos, current_cell)
                    stack.append(new_cell)
                next_positions.remove(pos)
            explored.add(current_cell.get_position())
    print("Maze Generation Complete.")
    return maze

def create_video_from_frames(path, out, frames_per_second, batch_size):
    image_path = path + 'images/'
    video_path = path + 'videos/'
    files = [f for f in os.listdir(image_path) if isfile(join(image_path, f))]
    files.sort(key = lambda x: x[5:-4])

    count = 0
    number_of_runs = len(files) // batch_size
    for b in range(0, number_of_runs+1):
        print("\nRunning Batch {} of {}.".format(b+1, number_of_runs+1))
        frame_array = []
        finish = batch_size * (b+1)
        if finish > len(files):
            finish = len(files)
        for i in tqdm(range(batch_size*b, finish)):
            filename = image_path + files[i]
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
        
        vid = cv2.VideoWriter(join(video_path,'video_batch_{}.mp4'.format(str(b+1).zfill(4))), cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, size)
        print("\nGenerating video file for batch {}.".format(b+1))
        for i in tqdm(range(len(frame_array))):
            vid.write(frame_array[i])
        vid.release()
        

    final_img = cv2.imread(image_path + files[-1])
    vid = cv2.VideoWriter(join(video_path,'video_batch_{}.mp4'.format(str(number_of_runs+2).zfill(4))), cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, size)
    for i in range(0,2*frames_per_second):
        vid.write(final_img)
    vid.release()
    
    video_files = [f for f in os.listdir(video_path) if isfile(join(video_path, f))]
    video_files.sort(key = lambda x: x[11:-4])
    videos = [VideoFileClip(join(video_path, i)) for i in video_files]
    final_video = concatenate_videoclips(videos)
    video_filename = out + '.mp4'
    final_video.write_videofile(video_filename)

def generate_maze(rows, columns, method, batch):
    path = './' + uuid.uuid4().hex + '/'
    os.mkdir(path)
    os.mkdir(path + 'images/')
    os.mkdir(path + 'videos/')
    print("")
    print('Creating folders in {} for aditional data strorage.'.format(path))
    print("Remember to Delete additional files after the process is complete.")
    print("")    
    init_maze = Maze(rows, columns)
    if method == "dfs":
        maze = dfs(init_maze, path + 'images/')
    elif method == "kruskals":
        maze = kruskals(init_maze, path + 'images/')
        
    video_filename = '{}_{}'.format(method, rows*columns)
    create_video_from_frames(path, video_filename, 24, batch)
    return maze

if __name__ == "__main__":
    size, batch, algo = get_arguments()
    MAX_ROW = 9*size
    MAX_COLUMN = 16*size
    maze = generate_maze(MAX_ROW, MAX_COLUMN, algo, batch)

