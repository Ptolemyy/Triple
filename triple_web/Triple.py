import numpy as np
import random
import math

class Triple:
    def __init__(self):
        super().__init__()
        self.board = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        self.board = np.reshape(self.board, 16)
        self.num_pool = np.array([1, 1])
        self.score = 0
    def log3(self, x):
        x = int(x)
        f = np.log(x)/np.log(3) if x > 0 else 0
        return f
    def possible_num(self):
        n = int(max(0,self.log3(np.max(self.board))-2))
        nums = [3 ** x for x in range(0,n+1)]
        return nums

    def update_pool(self):
        num = np.copy(self.possible_num())
        self.num_pool[0] = np.copy(self.num_pool[1])
        self.num_pool[1] = np.copy(num[random.randint(0,len(num)-1)])

    def search(self, pos, num):
        mapping = []
        fdirect = [-6,+1,+6,-1]
        rdirect = [-4,+1,+4,-1]
        x = np.reshape(self.board,(4,4))
        x = np.pad(x,pad_width=1,mode="constant",constant_values=0)
        x = np.reshape(x,36)

        inital_map = [0,0]
        inital_map[1] = pos
        mapping.append(inital_map)
        searching = True
        while searching:
            searching = False
            for i in mapping:
                for _ in range(0,4):
                    N0 = i[0]
                    pos1 = i[1]
                    pos0 = 7+pos1+2*(math.floor(pos1 / 4))
                    if N0 != 4:
                        direct = rdirect[N0]
                        if x[pos0+fdirect[N0]] == num and (pos1+direct not in [y[1] for y in mapping]):
                            new_map = [0,0]
                            new_map[1] = pos1+direct
                            searching = True
                            mapping.append(new_map)
                        i[0] = N0+1
        if len(mapping) >= 3:
            for i in mapping:
                self.board[i[1]] = 0
                self.place(pos,num*3)
                self.score += num*3
    
    def place(self, pos, num):
        if self.board[pos] == 0:
            self.board[pos] = num
            self.search(pos,num)
            self.score += 1

if __name__ == "__main__":
    gm = Triple()
    while True:
        print(np.reshape(gm.board,(4,4)))
        print(gm.num_pool)
        pos = int(input())
        gm.place(pos,gm.num_pool[0])
        gm.update_pool()
