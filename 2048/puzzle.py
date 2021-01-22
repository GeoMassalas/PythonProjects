import random


class puzzle:

    def __init__(self):
        self.state = self.generate_puzzle()
        self.valid = True

    def generate_puzzle(self):
        state = [[0 for i in range(4)] for j in range(4)]
        map1 = (0,0)
        map2 = (0,0)
        while map1 == map2:
            map1 = (random.randint(0,3), random.randint(0,3))
            map2 = (random.randint(0,3), random.randint(0,3))
        state[map1[0]][map1[1]] = 2 if random.random() < 0.8 else 4
        state[map2[0]][map2[1]] = 2 if random.random() < 0.8 else 4
        return state

    def reset(self):
        self.state = generate_puzzle()
        self.valid = True

    def get_state(self):
        return self.state

    def get_row(self, row):
        return self.state[row]

    def get_column(self, col):
        return [i[col] for i in self.state]

    def get_max_mun(self):
        return max(map(max, self.state))

    def get_score(self):
        return sum(sum(self.state,[]))

    def print_state(self):
        tmp = "\n"*60
        for i in range(4):
            for j in range(4):
                tmp += str(self.state[i][j]).rjust(5)
            tmp += "\n"
        print(tmp, end="", flush=True)

    def add_two(self):
        choices = []
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == 0:
                    choices.append((i,j))
        if len(choices) > 0:
            c = random.choice(choices)
            self.state[c[0]][c[1]] = 2
        else:
            self.valid = False

    def is_valid(self):
        return self.valid

    def clamp_lst(self, lst):
        new_lst = []
        j = 0
        while(j < len(lst)):
            if (j+1 < len(lst)) and (lst[j] == lst[j+1]):
                new_lst.append(lst[j]*2)
                j += 2
            else:
                new_lst.append(lst[j])
                j += 1
        new_lst += [0] * ( 4 - len(new_lst))
        return new_lst

    def move(self, position):
        tmp = []
        if position == "left":
            for i in range(4):
                row = [y for y in self.get_row(i) if y != 0]
                tmp.append(self.clamp_lst(row))
            self.state = tmp
                
        if position == "right":
            for i in range(4):
                row = [y for y in self.get_row(i) if y != 0]
                tmp.append(self.clamp_lst(row[::-1])[::-1])
            self.state = tmp
         
        if position == "up":
            temp = [["0" for x in range(4)] for y in range(4)] 
            for i in range(4):
                col = [y for y in self.get_column(i) if y != 0]
                tmp.append(self.clamp_lst(col))
            for i in range(4):
                for j in range(4):
                    temp[j][i] = tmp[i][j]
            self.state = temp

        if position == "down":
            temp = [["0" for x in range(4)] for y in range(4)] 
            for i in range(4):
                col = [y for y in self.get_column(i) if y != 0]
                tmp.append(self.clamp_lst(col[::-1])[::-1])
            for i in range(4):
                for j in range(4):
                    temp[j][i] = tmp[i][j]
            self.state = temp

        self.add_two()

if __name__ == "__main__":
    p = puzzle()
    p.print_state()
    choices = ["up", "down", "left", "right"]
    while(p.is_valid()):
        c = random.choice(choices)
        print("Moving " + c)
        p.move(c)
        p.print_state()
    print(p.get_metrics())
