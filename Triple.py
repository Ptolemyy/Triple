import numpy as np
import random
import math
board = np.array([[0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,0]])
board = np.reshape(board,16)
num_pool = np.array([1,1])

def log3(x):
    return np.log(x)/np.log(3)
def possible_num():
    n = int(max(0,log3(max(board))-2))
    nums = [3**x for x in range(0,n+1)]
    return nums
def update_pool():
    num = possible_num()
    num_pool[1] = np.copy(num_pool[0])
    num_pool[0] = np.copy(num[random.randint(0,len(num)-1)])
def search(pos,num):
    mapping = []
    fdirect = [-6,+1,+6,-1]
    rdirect = [-4,+1,+4,-1]
    x = np.reshape(board,(4,4))
    x = np.pad(x,pad_width=1,mode="constant",constant_values=0)
    x = np.reshape(x,(36))
    
    inital_map = [0,0]
    inital_map[1] = pos
    mapping.append(inital_map)
    searching = True
    while searching:
        searching = False
        for i in mapping:
            for j in range(0,4):
                N0 = i[0]
                pos1 = i[1]
                pos0 = 7+pos1+2*(math.floor((pos1)/4))
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
            board[i[1]] = 0
            place(pos,num*3)
def place(pos,num):
    if board[pos] == 0:
        board[pos] = num 
        search(pos,num)
        update_pool()
def set_board(x):
    board = np.copy(x)
    board = np.reshape(board,16)
def set_pool(x):
    num_pool = np.copy(x)
if __name__ == "__main__":
    while True:
        print(np.reshape(board,(4,4)))
        print(num_pool)
        pos = int(input())
        place(pos,num_pool[1])
