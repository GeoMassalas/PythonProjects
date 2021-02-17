import numpy as np
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import argparse
import os
import sys
from tqdm import tqdm
from os.path import isfile, join
from collections import deque
import heapq
from random import choice, shuffle
import uuid
import time
from subprocess import call
from disjointset import DisjointSet

# TODO:
# add comments
# create a better structure than a single 600 line file

# Validation lists
GENERATION_ALGORITHMS = ['prims', 'dfs', 'kruskals']
SOLUTION_ALGORITHMS = ['astar', 'bfs', 'dfs']
VIDEO_GENERATION_OPTIONS = ['both', 'gen', 'solve']
RESOLUTIONS = [1080, 1050, 960, 900, 800, 768, 720, 648, 600, 576, 480]
ASPECT_RATIOS = ["4:3", "16:10", "16:9"]

class Maze:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = [['rlud' for j in range(self.width)] for i in range(self.height)]
        self.path = []
        self.explored = []


        self.cell_size = min(int(RESOLUTION[1] / height), int(RESOLUTION[0] / width))
        self.offset_w = (RESOLUTION[0] - self.width *  self.cell_size) // 2
        self.offset_h = (RESOLUTION[1] - self.height * self.cell_size) // 2
        print(self.offset_w, self.offset_h)
        self.color = (0,0,0)
        self.path_color = (212, 81, 63)
        self.explored_color = (171, 68, 166)
        self.background = 255
        self.line_width = LINE_WIDTH

    def get_number_of_cells(self):
        return self.height * self.width

    def get_rows(self):
        return self.height

    def get_cols(self):
        return self.width

    def get_avaliable_moves(self, position):
        r, c = position
        moves = []
        if "u" not in self.grid[r][c]:
            moves.append((r-1,c))
        if "r" not in self.grid[r][c]:
            moves.append((r,c+1))
        if "d" not in self.grid[r][c]:
            moves.append((r+1,c))
        if "l" not in self.grid[r][c]:
            moves.append((r,c-1))
        return moves

    def add_to_path(self, position):
        self.path.append(position)

    def add_to_explored(self, position):
        self.explored.append(position)

    def clear_path(self):
        self.path = []

    def clear_explored(self):
        self.explored = []

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

    def draw_explored(self, image):
        for cell_p in self.explored:
            p1 = self.offset_w + cell_p[1] * self.cell_size, self.offset_h + cell_p[0] * self.cell_size
            p2 = self.offset_w + (cell_p[1] + 1 ) * self.cell_size, self.offset_h + (cell_p[0] + 1) * self.cell_size,
            cv2.rectangle(image, pt1=p1, pt2=p2, color=self.explored_color, thickness=-1)
        return image


    def draw_path(self, image):
        for cell_p in self.path:
            p1 = self.offset_w + cell_p[1] * self.cell_size, self.offset_h + cell_p[0] * self.cell_size
            p2 = self.offset_w + (cell_p[1] + 1 ) * self.cell_size, self.offset_h + (cell_p[0] + 1) * self.cell_size,
            cv2.rectangle(image, pt1 = p1, pt2= p2, color=self.path_color, thickness=-1)
        return image

    def paint_cell(self, img, r, c):
        corners = {"nw": (self.offset_w+c*self.cell_size, self.offset_h+r*self.cell_size),
                "sw": (self.offset_w+c*self.cell_size, self.offset_h+(r+1)*self.cell_size),
                "se":(self.offset_w+(c+1)*self.cell_size, self.offset_h+(r+1)*self.cell_size),
                "ne":(self.offset_w+(c+1)*self.cell_size, self.offset_h+r*self.cell_size)}
        for char in self.grid[r][c]:
            if char == 'r':
                img = cv2.line(img, corners["ne"], corners["se"], self.color, self.line_width)
            if char == 'l':
                img = cv2.line(img, corners["nw"], corners["sw"], self.color, self.line_width)
            if char == 'u':
                img = cv2.line(img, corners["nw"], corners["ne"], self.color, self.line_width)
            if char == 'd':
                img = cv2.line(img, corners["sw"], corners["se"], self.color, self.line_width)

        return img

    def generate_image_state(self):
        img = np.zeros([RESOLUTION[1], RESOLUTION[0], 3], dtype=np.uint8 )
        img.fill(self.background)
        if len(self.explored) > 0:
            img = self.draw_explored(img)
        if len(self.path) > 0:
            img = self.draw_path(img)
        img = cv2.line(img, (0,0), (RESOLUTION[0],0), self.color, self.offset_h*2 + 1)
        img = cv2.line(img, (0,0), (0,RESOLUTION[1]), self.color, self.offset_w*2 + 1)
        img = cv2.line(img, (RESOLUTION[0], RESOLUTION[1]), (RESOLUTION[0], 0), self.color, self.offset_w*2 + 1)
        img = cv2.line(img, (RESOLUTION[0], RESOLUTION[1]), (0, RESOLUTION[1]), self.color, self.offset_h*2 + 1)
        return img

    def get_canvas(self):
        img = self.generate_image_state()
        for r in range(0, self.height):
            for c in range(0, self.width):
                img = self.paint_cell(img, r, c)
        return img

    def print_canvas(self, name):
        img = self.get_canvas()
        cv2.imwrite(name + ".png", img)


class Cell:

    def __init__(self, position, parent=None, depth=0, cost=0):
        self.row = position[0]
        self.column = position[1]
        self.parent = parent
        self.depth = depth
        self.est_cost = cost + depth

    def __lt__(self, other):
        return self.est_cost < other.est_cost

    def get_position(self):
        return (self.row, self.column)
    
    def get_parent(self):
        return self.parent

    def get_depth(self):
        return self.depth

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
    parser.add_argument('-l', '--line-width',
            dest='line_width', type=int,
            help="Sudoku line width. Default: 1")
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
    if not args.line_width:
        args.line_width = 1
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

def kruskals(maze, generate_images):
    walls = get_walls(maze)
    shuffle(walls)
    cells = [(r,c) for r in range(maze.get_rows()) for c in range(maze.get_cols())]
    dset = DisjointSet(cells)
    batch = 0
    count = 0
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    print("Generating Maze...")
    start = time.time()
    while len(walls) > 0:
        wall = walls.pop()
        if dset.union(wall[0], wall[1]) == False:
            maze.connect_cells(wall[0], wall[1])
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                    image_array.fill(0)
    if generate_images:
        final_image = maze.get_canvas()
        for i in range(2*FPS):
            image_array[count] = final_image
            count += 1
        indices = range(count)
        create_video_from_frames(batch, image_array[indices])
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

def prims(maze, generate_images):
    cell = (0,0)
    batch = 0
    count = 0
    explored = set([cell])
    walls = get_walls_p(cell)
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    print("Generating Maze...")
    start = time.time()
    while(len(walls) > 0):
        shuffle(walls)
        wall = walls.pop()
        if len(set([wall[1]]) - explored) == 1:
            maze.connect_cells(wall[0], wall[1])
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                image_array[count] = maze.get_canvas()
            new_walls = get_walls_p(wall[1])
            for w in new_walls:
                walls.append(w)
            explored.add(wall[1])
    if generate_images:
        final_image = maze.get_canvas()
        for i in range(2*FPS):
            image_array[count] = final_image
            count += 1
        indices = range(count)
        create_video_from_frames(batch, image_array[indices])
    el_time = time.time() - start
    print("Maze Generation Complete in {}.".format(el_time))
    return maze

def dfs(maze, generate_images):
    parent_cell = Cell((0,0))
    init_cell = Cell((0,0), parent_cell)
    explored = set()
    stack = deque()
    stack.append(init_cell)
    batch = 0
    count = 0
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    print("Generating Maze...")
    start = time.time()
    while(len(stack) > 0):
        current_cell = stack.pop()
        if current_cell.get_position() not in explored:
            maze.connect_cells(current_cell.get_position(), current_cell.parent.get_position())
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                    image_array.fill(0)
            next_positions = current_cell.get_neighbors()
            while(len(next_positions) > 0):
                pos = choice(next_positions)
                if pos not in explored:
                    new_cell = Cell(pos, current_cell)
                    stack.append(new_cell)
                next_positions.remove(pos)
            explored.add(current_cell.get_position())
    if generate_images:
        final_image = maze.get_canvas()
        for i in range(2*FPS):
            image_array[count] = final_image
            count += 1
        indices = range(count)
        create_video_from_frames(batch, image_array[indices])
    el_time = time.time() - start
    print("Maze Generation Complete in {}.".format(el_time))
    return maze

def dfs_solve(maze, generate_images):
    init_cell = Cell((0,0))
    target = (maze.get_rows() - 1, maze.get_cols() - 1)
    explored = set()
    stack = deque()
    stack.append(init_cell)
    batch = 0
    count = 0
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    print("Solving Maze...")
    start = time.time()
    while(len(stack) > 0):
        current_cell = stack.pop()
        if current_cell.get_position() == target:
            el_time = time.time() - start
            print("Maze Solution Complete in {}.".format(el_time))
            maze.add_to_explored(target)
            solve_path = []
            if generate_images:
                final_image = maze.get_canvas()
                image_array[count] = final_image
                count += 1
                indices = range(count)
                create_video_from_frames(batch, image_array[indices])
                batch += 1
            while current_cell.get_parent() != None:
                solve_path.append(current_cell.get_position())
                current_cell = current_cell.get_parent()
            solve_path.append(current_cell.get_position())
            return solve_path[::-1], batch
        elif current_cell.get_position() not in explored:
            next_positions = maze.get_avaliable_moves(current_cell.get_position())
            for pos in next_positions: 
                if pos not in explored:
                    new_cell = Cell(pos, current_cell)
                    stack.append(new_cell)
            explored.add(current_cell.get_position())
            maze.add_to_explored(current_cell.get_position())
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                    image_array.fill(0)

def bfs_solve(maze, generate_images):
    cell = Cell((0,0))
    target = (maze.get_rows() - 1, maze.get_cols() - 1)
    q = deque()
    q.append(cell)
    explored = set()
    batch = 0
    count = 0
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    start = time.time()
    print("Solving maze...")
    while(len(q) > 0):
        current_cell = q.popleft()
        pos = current_cell.get_position()
        if pos == target:    
            el_time = time.time() - start
            print("Maze Solution Complete in {}.".format(el_time))
            maze.add_to_explored(target)
            solve_path = []
            if generate_images:
                final_image = maze.get_canvas()
                image_array[count] = final_image
                count += 1
                indices = range(count)
                create_video_from_frames(batch, image_array[indices])
                batch += 1
            while current_cell.get_parent() != None:
                solve_path.append(current_cell.get_position())
                current_cell = current_cell.get_parent()
            solve_path.append(current_cell.get_position())
            return solve_path[::-1], batch
        else:
            positions = maze.get_avaliable_moves(pos)
            for i in range(len(positions)):
                if positions[i] not in explored:
                    q.append(Cell(positions[i], current_cell))
                    explored.add(positions[i])
                    maze.add_to_explored(positions[i])
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                    image_array.fill(0)

def manhatan_distance(pos, target):
    return (target[0] - pos[0]) + (target[1] - pos[1])

def solve_astar(maze, generate_images):
    target = (maze.get_rows() - 1, maze.get_cols() - 1)
    cell = Cell((0,0), None, depth=0, cost=manhatan_distance((0,0), target))
    heap = []
    heapq.heapify(heap)
    heapq.heappush(heap, cell)
    explored = set()
    batch = 0
    count = 0
    image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    start = time.time()
    print("Solving maze...")
    while(len(heap) > 0):
        current_cell = heapq.heappop(heap)
        if current_cell.get_position() == target:
            el_time = time.time() - start
            print("Maze Solution Complete in {}.".format(el_time))
            maze.add_to_explored(target)
            solve_path = []
            if generate_images:
                final_image = maze.get_canvas()
                image_array[count] = final_image
                count += 1
                indices = range(count)
                create_video_from_frames(batch, image_array[indices])
                batch += 1
            while current_cell.get_parent() != None:
                solve_path.append(current_cell.get_position())
                current_cell = current_cell.get_parent()
            solve_path.append(current_cell.get_position())
            return solve_path[::-1], batch
        else:
            cells = maze.get_avaliable_moves(current_cell.get_position())   
            for cell in cells:
                if cell not in explored:
                    heapq.heappush(heap, Cell(cell, current_cell, current_cell.get_depth() + 1, manhatan_distance(cell, target)))
                    explored.add(cell)
                    maze.add_to_explored(cell)
            if generate_images:
                image_array[count] = maze.get_canvas()
                count += 1
                if count >= BATCH_SIZE:
                    create_video_from_frames(batch, image_array)
                    batch += 1
                    count = 0
                    image_array.fill(0)

def create_video_from_frames(batch, frame_array):
    video_name = 'video_batch_{}.mp4'.format(str(batch+1).zfill(4))
    print("Generating {}".format(video_name))
    vid = cv2.VideoWriter(join(PATH, video_name), cv2.VideoWriter_fourcc(*'mp4v'), FPS, RESOLUTION)
    pbar = tqdm(total=len(frame_array))
    for frame in frame_array:
        vid.write(frame)
        pbar.update()
    pbar.close()
    vid.release()

def create_video(out):
    video_files = [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
    video_files.sort(key = lambda x: x[11:-4])
    videos = [VideoFileClip(join(PATH, i)) for i in video_files]
    final_video = concatenate_videoclips(videos)
    video_filename = out + '.mp4'
    final_video.write_videofile(video_filename)

def generate_maze(rows, columns, method, generate_video):
    if generate_video:
        os.mkdir(PATH)
        print("")
        print('Creating folder {} for aditional data strorage.'.format(PATH))
        print("Remember to Delete additional files if the script fails.")
        print("")
    init_maze = Maze(rows, columns)
    if method == "dfs":
        maze = dfs(init_maze, generate_video)
    elif method == "kruskals":
        maze = kruskals(init_maze, generate_video)
    elif method == "prims":
        maze = prims(init_maze, generate_video)

    if generate_video:
        video_filename = '{}_{}_{}'.format(method, rows*columns, str(RESOLUTION[1]) + "p")
        create_video(video_filename)
        print("Deleting extra files...")
        call('rm -rf ' + PATH, shell=True)
    return maze

def solve_maze(maze, method, generate_video):
    if generate_video:
        os.mkdir(PATH)
        print("")
        print('Creating folder {} for aditional data strorage.'.format(PATH))
        print("Remember to Delete additional files if the script fails.")
        print("")
    if method == "dfs":
        path, batch = dfs_solve(maze, generate_video)
    elif method == "bfs":
        path, batch = bfs_solve(maze, generate_video)
    elif method == "astar":
        path, batch = solve_astar(maze, generate_video)

    if generate_video:
        count = 0
        image_array = np.zeros((BATCH_SIZE, RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        for i in path:
            if count >= BATCH_SIZE:
                create_video_from_frames(batch, image_array)
                batch += 1
                count = 0
                image_array.fill(0)
            maze.add_to_path(i)
            image_array[count] = maze.get_canvas()
            count += 1
        final_image = maze.get_canvas()
        for i in range(2*FPS):
            image_array[count] = final_image
            count += 1
        indices = range(count)
        create_video_from_frames(batch, image_array[indices])
        
        video_filename = '{}_solve_{}_{}'.format(method, maze.get_number_of_cells(), str(RESOLUTION[1]) + "p")
        create_video(video_filename)
        print("Deleting extra files...")
        call('rm -rf ' + PATH, shell=True)
    else:
        for step in path:
            maze.add_to_path(step)
    
if __name__ == "__main__":
    size, batch, algo, gen_vid, solve_vid, fps, res, aspr = get_arguments()
    MAX_ROW = size
    MAX_COLUMN = int(size*aspr[0]/aspr[1])
    RESOLUTION = res
    FPS = fps
    BATCH_SIZE = batch
    LINE_WIDTH = 2
    PATH = './' + uuid.uuid4().hex + '/'
    algo = "kruskals"
    s_algo = "astar"
    gen_vid = False
    solve_vid = False
    maze = generate_maze(MAX_ROW, MAX_COLUMN, algo, gen_vid)
    maze.print_canvas("maze")
    solve_maze(maze, s_algo, solve_vid)
    maze.clear_explored()
    maze.print_canvas("solution")
