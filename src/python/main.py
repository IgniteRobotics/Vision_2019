import numpy as np

FRAMES = 10
VALUES = 4


#pushes array v onto 2d array a,
#then slices a to maintain its original size
def push(a, v):
    i,j = a.shape
    print('a:',a,'v:',v,'i:',i,'j:',j)
    a = np.vstack((a,v))
    a = a[-i:]
    return a


#column-wise average of a 2d array
def avg(a):
    return a.mean(axis=0)


###### Main ####

## create a 2d array of zeros
a = np.zeros((FRAMES,VALUES), dtype=np.float32)

print('array', a, 'len: ',a.size)

for i in range(100):
    #1d array of randoms
    b = np.random.rand(VALUES)
    a = push(a,b)
    print('index:',i,'values:',a, 'averages:', avg(a))


