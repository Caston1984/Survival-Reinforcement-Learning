import random

import pandas as pd

ls = []
ls1 = []
ls2 = []

F = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
W = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

for x in X:
    for y in Y:
        a = [x, y]
        ls.append(a)

for w in W:
    for element in ls:
        ls1.append([w] + element)

for f in F:
    for element in ls1:
        ls2.append([f] + element)

print(ls2)
print(len(ls2))
print(ls2.index([4, 6, 5, 5]))

def terminal_state(row):
    flag = False
    if (row[0] == 0 or row[0] == 4) or (row[1] == 1 or row[1] == 6):
        flag = True

    return flag

for i in range(100):
    state = random.choice(ls2)
    print('Original state', state)
    while terminal_state(state):
        print('TRY AGAIN')
        state = random.choice(ls2)
        print(('Revised state', state))


state = [2, 1, 5, 1]
print(terminal_state(state))