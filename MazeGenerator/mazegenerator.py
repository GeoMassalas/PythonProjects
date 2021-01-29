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
import time
from subprocess import call
from disjointset import DisjointSet
# TODO:
# 1)revisit the write images to disk and then load them again process to create the video in batches once the maze is fully generated
# I could just save them in a variable and when i get length = num of batches create the video_batch_x
# 2) start working on solving algorithms

# Validation lists
GENERATION_ALGORITHMS = ['prims', 'dfs', 'kruskals']
SOLUTION_ALGORITHMS = ['astar', 'bfs', 'dfs']
VIDEO_GENERATION_OPTIONS = ['both', 'gen', 'solve']
RESOLUTIONS = [1080, 1050, 960, 900, 800, 768, 720, 648, 600, 576, 480]
ASPECT_RATIOS = ["4:3", "16:10", "16:9"]

# Variables needed
IMAGE_TYPE = '.bmp' # bmp for speed // png for size
WALL_COLOR = (0, 0, 0)
BACKGROUND_COLOR = 255
OFFSET = 2
CELL_SIZE = 16

class Maze:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = [['rlud' for j in range(self.width)] for i in range(self.height)]

        self.cell_size = CELL_SIZE
        self.color = WALL_COLOR
        self.background = BACKGROUND_COLOR
        self.offset = OFFSET

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
        resized_img = cv2.resize(img, RESOLUTION)
        cv2.imwrite(name + IMAGE_TYPE, resized_img)


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
            help='Height of the maze in cells. Range: 10-100')
    parser.add_argument('-b', '--batch-size',
            dest='batch_size', type=int,
            help='Batch size for the video generation. Default Batch: 1000.')
    parser.add_argument('-g', '--generator', 
            dest='gen_algo', type=str,
            help="Algorithm to use in maze generation.\nOptions: dfs, kruskals, prims\nDefault option: dfs")
    parser.add_argument('-v', '--video', 
            dest='video', type=str,
            help="Generate video for maze generation and/or solution.\nOptions: gen, solve, both")
    parser.add_argument('-f', '--fps', 
            dest='fps', type=int,
            help="Frames per second on output video. This effects how fast the algorithm is displayed.\nRange:10-60\nDefault:24")
    parser.add_argument('-r', '--resolution', 
            dest='resolution', type=int,
            help="Resolution on output video.\nOptions: 1080, 1050, 960, 900, 800, 768, 720, 648, 600, 576, 480\nDefault:720")
    parser.add_argument('-a', '--aspect-ratio', 
            dest='ar', type=str,
            help="Aspect ratio on output video. Options: 4:3, 16:10, 16:9\nDefault: 16:9")
    
    args = parser.parse_args()
    
    # Maze Size Validation
    if (args.size < 10) | (args.size > 99):
        parser.error("Size needs to be between 10 and 99.")
    else:
        size = args.size*args.size
    # Batch Size Validation
    if not args.batch_size:
        args.batch_size = 1000
    else:
        if (args.batch_size < 100) | (args.batch_size > size):
            parser.error("Batch size needs to be between 100 and maze size(Current Size: {}).".format(size))
    # Generation algorithm validation
    if args.gen_algo:
        if args.gen_algo not in GENERATION_ALGORITHMS:
            parser.error("Invalid generation algorithm. Valid options: dfs, kruskals, prims.")
    else:
        args.gen_algo = 'dfs'
    # Video generation options validation
    gen_vid = False
    solve_vid = False
    if args.video:
        if args.video not in VIDEO_GENERATION_OPTIONS:
            parser.error("Invalid video generation option. Valid options: solve, gen, both.")
        if args.video != 'gen':
            solve_vid = True
        if args.video != 'solve':
            gen_vid = True
    # Frames per second validation
    if args.fps:
        if (args.fps < 10) | (args.fps > 60):
            parser.error("Invalid frame rate option. Valid range is 10 to 60.")
    else:
        args.fps = 24
    # Resolution validation
    if args.resolution:
        if args.resolution not in RESOLUTIONS:
            parser.error("Invalid resolution. Valid options: 1080, 1050, 960, 900, 800, 768, 720, 648, 600, 576, 480")
    else:
        args.resolution = 720
    # Aspect Ration validation
    if args.ar:
        if args.ar not in ASPECT_RATIOS:
            parser.error("Invalid aspect ration. Valid options: 4:3, 16:9, 16:10")
        else:
            ar = [int(c) for c in args.ar.split(":")]
    else:
        ar = [16, 9]
    res = (int(args.resolution * ar[0] / ar[1]), args.resolution)
    return args.size, args.batch_size, args.gen_algo, gen_vid, solve_vid, args.fps, res, ar

def get_walls(maze):
    walls = []
    for r in range(0,maze.get_rows()):
        for c in range(0,maze.get_cols()):
            if r < maze.get_rows() - 1:
                walls.append(((r, c), (r+1, c)))
            if c < maze.get_cols() - 1:
                walls.append(((r, c), (r, c+1)))
    return walls

def kruskals(maze, image_folder, generate_images):
    walls = get_walls(maze)
    shuffle(walls)
    cells = [(r,c) for r in range(maze.get_rows()) for c in range(maze.get_cols())]
    dset = DisjointSet(cells)
    count = 0
    print("Generating Maze...")
    start = time.time()
    pbar = tqdm(total=len(walls))
    while len(walls) > 0:
        wall = walls.pop()
        if dset.union(wall[0], wall[1]) == False:
            maze.connect_cells(wall[0], wall[1])
            if generate_images:
                maze.print_canvas('{}image{}'.format(image_folder, str(count).zfill(len(str(maze.get_number_of_cells())))))
                count += 1
        pbar.update()
    pbar.refresh()
    maze.print_canvas('kruskal_test')
    el_time = time.time() - start
    print("Maze Generation Complete in {}.".format(el_time))
    return maze

def get_walls_p(cell):
    res = []
    if cell[0] > 0:
        res.append(((cell[0], cell[1]), (cell[0] - 1, cell[1])))
    if cell[1] > 0:
        res.append(((cell[0], cell[1]), (cell[0], cell[1] - 1)))
    if cell[0] < MAX_ROW - 1:
        res.append(((cell[0], cell[1]), (cell[0] + 1, cell[1])))
    if cell[1] < MAX_COLUMN - 1:
        res.append(((cell[0], cell[1]), (cell[0], cell[1] + 1)))
    return res

def prims(maze, image_folder, generate_images):
    cell = (0,0)
    count = 0
    explored = set([cell])
    walls = get_walls_p(cell)
    print("Generating Maze...")
    start = time.time()
    pbar = tqdm(total=maze.get_number_of_cells())
    while(len(walls) > 0):
        shuffle(walls)
        wall = walls.pop()
        if len(set([wall[1]]) - explored) == 1:
            maze.connect_cells(wall[0], wall[1])
            if generate_images:
                maze.print_canvas('{}image{}'.format(image_folder, str(count).zfill(len(str(maze.get_number_of_cells())))))
                count += 1
            pbar.update()
            new_walls = get_walls_p(wall[1])
            for w in new_walls:
                walls.append(w)
            explored.add(wall[1])
    pbar.close()
    el_time = time.time() - start
    print("Maze Generation Complete in {}.".format(el_time))
    return maze

def dfs(maze, image_folder, generate_images):
    parent_cell = Cell((0,0))
    init_cell = Cell((0,0), parent_cell)
    explored = set()
    stack = deque()
    stack.append(init_cell)
    count = 0
    print("Generating Maze...")
    start = time.time()
    pbar = tqdm(total=maze.get_number_of_cells())
    while(len(stack) > 0):
        current_cell = stack.pop()
        if current_cell.get_position() not in explored:
            maze.connect_cells(current_cell.get_position(), current_cell.parent.get_position())
            if generate_images:
                maze.print_canvas('{}image{}'.format(image_folder, str(count).zfill(len(str(maze.get_number_of_cells())))))
                count += 1
            pbar.update()
            next_positions = current_cell.get_neighbors()
            while(len(next_positions) > 0):
                pos = choice(next_positions)
                if pos not in explored:
                    new_cell = Cell(pos, current_cell)
                    stack.append(new_cell)
                next_positions.remove(pos)
            explored.add(current_cell.get_position())
    #pbar.refresh()
    pbar.close()
    el_time = time.time() - start
    print("Maze Generation Complete in {}.".format(el_time))
    return maze

def create_video_from_frames(path, out, frames_per_second, batch_size):
    image_path = path + 'images/'
    video_path = path + 'videos/'
    files = [f for f in os.listdir(image_path) if isfile(join(image_path, f))]
    files.sort(key = lambda x: x[5:-4])

    count = 0
    number_of_runs = len(files) // batch_size
    for b in range(0, number_of_runs+1):
        print("Running Batch {} of {}.".format(b+1, number_of_runs+1))
        frame_array = []
        finish = batch_size * (b+1)
        if finish > len(files):
            finish = len(files)
        
        pbar = tqdm(total=finish-(batch_size*b))
        for i in range(batch_size*b, finish):
            filename = image_path + files[i]
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
            pbar.update()
        pbar.close()
        vid = cv2.VideoWriter(join(video_path,'video_batch_{}.mp4'.format(str(b+1).zfill(4))), cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, size)
        print("Generating video file for batch {}.".format(b+1))
        pbar = tqdm(total=len(frame_array))
        for i in range(len(frame_array)):
            vid.write(frame_array[i])
            pbar.update()
        pbar.close()
        print("")
        vid.release()
        

    final_img = cv2.imread(image_path + files[-1])
    vid = cv2.VideoWriter(join(video_path,'video_batch_{}.mp4'.format(str(number_of_runs+2).zfill(4))), cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, size)
    for i in range(0,2*frames_per_second):
        vid.write(final_img)
    print("")
    vid.release()
    
    video_files = [f for f in os.listdir(video_path) if isfile(join(video_path, f))]
    video_files.sort(key = lambda x: x[11:-4])
    videos = [VideoFileClip(join(video_path, i)) for i in video_files]
    final_video = concatenate_videoclips(videos)
    video_filename = out + '.mp4'
    final_video.write_videofile(video_filename)

def generate_maze(rows, columns, method, batch, fps, generate_video):
    path = './' + uuid.uuid4().hex + '/'
    if generate_video:
        os.mkdir(path)
        os.mkdir(path + 'images/')
        os.mkdir(path + 'videos/')
        print("")
        print('Creating folders in {} for aditional data strorage.'.format(path))
        print("Remember to Delete additional files if the script fails.")
        print("")
    init_maze = Maze(rows, columns)
    if method == "dfs":
        maze = dfs(init_maze, path + 'images/', generate_video)
    elif method == "kruskals":
        maze = kruskals(init_maze, path + 'images/', generate_video)
    elif method == "prims":
        maze = prims(init_maze, path + 'images/', generate_video)
        
    if generate_video:
        video_filename = '{}_{}_{}'.format(method, rows*columns, str(RESOLUTION[1]) + "p")
        create_video_from_frames(path, video_filename, fps, batch)
        print("Deleting extra files...")
        call('rm -rf ' + path, shell=True)
    return maze

if __name__ == "__main__":
    size, batch, algo, gen_vid, solve_vid, fps, res, aspr= get_arguments()
    MAX_ROW = size
    MAX_COLUMN = int(size*aspr[0]/aspr[1])
    RESOLUTION = res
    algo = "prims"
    gen_vid = True
    maze = generate_maze(MAX_ROW, MAX_COLUMN, algo, batch, fps, gen_vid)
    maze.print_canvas("maze")
