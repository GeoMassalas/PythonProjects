import sys
import time

def generate_warnsdorff(n):
    return [[get_number_of_moves(r, c) for c in range(n)] for r in range(n)]

def build_init_board(n):
    return [["00" for j in range(n)] for i in range(n)]

def print_board():
    print("-----------------")
    for r in range(n):
        print(board[r])

def get_number_of_moves(r, c):
    count = 0
    if (r-2 >= 0) and (c-1 >= 0):
        count += 1
    if (r-1 >= 0) and (c-2 >= 0):
        count += 1
    if (r-2 >= 0) and (c+1 < n):
        count += 1
    if (r-1 >= 0) and (c+2 < n):
        count += 1
    if (r+2 < n) and (c-1 >= 0):
        count += 1
    if (r+1 < n) and (c-2 >= 0):
        count += 1
    if (r+2 < n) and (c+1 < n):
        count += 1
    if (r+1 < n) and (c+2 < n):
        count += 1
    return count

def get_moves(r, c):
    moves = []
    if (r-2 >= 0) and (c-1 >= 0) and (board[r-2][c-1] == "00"):
        moves.append((r-2, c-1, warnsdorff[r-2][c-1]))
    if (r-1 >= 0) and (c-2 >= 0) and (board[r-1][c-2] == "00"):
        moves.append((r-1, c-2, warnsdorff[r-1][c-2]))
    if (r-2 >= 0) and (c+1 < n) and (board[r-2][c+1] == "00"):
        moves.append((r-2, c+1, warnsdorff[r-2][c+1]))
    if (r-2 >= 0) and (c+2 < n) and (board[r-1][c+2] == "00"):
        moves.append((r-1, c+2, warnsdorff[r-1][c+2]))
    if (r+2 < n) and (c-1 >= 0) and (board[r+2][c-1] == "00"):
        moves.append((r+2, c-1, warnsdorff[r+2][c-1]))
    if (r+1 < n) and (c-2 >= 0) and (board[r+1][c-2] == "00"):
        moves.append((r+1, c-2, warnsdorff[r+1][c-2]))
    if (r+2 < n) and (c+1 < n) and (board[r+2][c+1] == "00"):
        moves.append((r+2, c+1, warnsdorff[r+2][c+1]))
    if (r+1 < n) and (c+2 < n) and (board[r+1][c+2] == "00"):
        moves.append((r+1, c+2, warnsdorff[r+1][c+2]))
    moves.sort(key=lambda x:x[2])
    return moves

def solve(r, c, depth=1):
    board[r][c] = str(depth).zfill(2)
    if depth==n*n:
        print_board()
        elapsed = time.time() - start
        print("----------------------")
        print("Solution for a {}x{} board found after {} Seconds.".format(n,n,"{:.5f}".format(elapsed)))
        sys.exit()
    else:
        possible_moves = get_moves(r,c)
        for move in possible_moves:
            solve(move[0], move[1], depth + 1)
            board[move[0]][move[1]] = "00"

if __name__ == "__main__":
    n = 8
    warnsdorff = generate_warnsdorff(n)
    board = build_init_board(n)
    start = time.time()
    solve(0,0)
