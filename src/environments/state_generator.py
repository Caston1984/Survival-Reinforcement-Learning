import pandas as pd
import numpy as np


# Python program to print all permutations using
# Heap's algorithm

# Generating permutation using Heap Algorithm
def heapPermutation(a, size):
    # if size becomes 1 then prints the obtained
    # permutation
    if size == 1:
        print(a)
        return

    for i in range(size):
        heapPermutation(a, size - 1)

        # if size is odd, swap 0th i.e (first)
        # and (size-1)th i.e (last) element
        # else If size is even, swap ith
        # and (size-1)th i.e (last) element
        if size & 1:
            a[0], a[size - 1] = a[size - 1], a[0]
        else:
            a[i], a[size - 1] = a[size - 1], a[i]



# Driver code
a = [1, 2, 3]
# n = len(a)
# heapPermutation(a, n)

ls = [[]]

def _heap_perm_(n, A):
    if n == 1: yield A
    else:
        for i in range(n-1):
            for hp in _heap_perm_(n-1, A): yield hp
            j = 0 if (n % 2) == 1 else i
            A[j],A[n-1] = A[n-1],A[j]
        for hp in _heap_perm_(n-1, A): yield hp


# Python function to print permutations of a given list
def permutation(lst):
    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []

    # If there is only one element in lst then, only
    # one permutation is possible
    if len(lst) == 1:
        return [lst]

    # Find the permutations for lst if there are
    # more than 1 characters

    l = []  # empty list that will store current permutation

    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
        m = lst[i]

        # Extract lst[i] or m from the list.  remLst is
        # remaining list
        remLst = lst[:i] + lst[i + 1:]

        # Generating all permutations where m is first
        # element
        for p in permutation(remLst):
            l.append([m] + p)
    return l

ls = []
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
F = [0, 1, 2, 3, 4]
#W = ['a', 'b', 'c']
W = [1, 2, 3, 4, 5, 6]
for f in F:
    for w in W:
        for x in X:
            for y in Y:
                a = [f, w, x, y]
                n = len(a)
                ls.extend(permutation(a))



print(len(ls))
res = []
[res.append(x) for x in ls if x not in res]
print(len(res))
df = pd.DataFrame(res)
print(df)
print(res.index([7, 5, 6, 1]))