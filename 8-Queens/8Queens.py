import sys

def check_column(column):
    for i in range(0,8):
        if board[i][column] == 1:
            return False
    return True

def check_row(row):
    for i in range(0,8):
        if board[row][i] == 1:
            return False
    return True

def check_diagonal(row, column):
    minimum = min(row,column)
    diff = abs(row-column)
    s = row + column

    r = row-minimum
    c = column-minimum
    while((r != 8) & (c != 8)):
        if board[r][c] == 1:
            return False
        r += 1
        c += 1
    if s < 8:
        c = 0
        r = s
    else:
        c = s % 7
        r = 7
    while((r >= 0) & (c < 8)):
        if board[r][c] == 1:
            return False
        r -= 1
        c += 1
    return True


def generate_board():
    return [[0 for i in range(8)] for j in range(8)]

def print_board():
    print("-----------------------")
    for i in range(0,8):
        print(board[i])
    print("----------------------")

def solve(depth=0):
    if depth==8:
        if board not in solutions:
            arr = [[board[j][i] for i in range(8)] for j in range(8)]
            solutions.append(arr)
            prompt = ""
            print("Solution Found!")
            print_board()
            while (prompt != "y") & (prompt != "Y"):
                prompt = input(str(len(solutions)) + " unique solution(s) found. Need more? (y/n) ")
                if (prompt == "n") | (prompt == "N"):
                    print("Exiting.")
                    sys.exit()
    else:
        for r in range(8):
            for c in range(8):
                if check_column(c) & check_row(r) & check_diagonal(r, c):
                    board[r][c] = 1
                    solve(depth+1)
                    board[r][c] = 0

if __name__ == "__main__":
    solutions = []
    board = generate_board()
    solve()
