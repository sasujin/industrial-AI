import numpy as np

# 2.3.3 가중치와 편향 구현하기
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

def NAND(x1, x2): 
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # AND와는 가중치(w, b)만 다르다
    b = 0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2): 
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # AND와는 가중치(w, b)만 다르다
    b = -0.2
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

# 2.5.2 XOR 게이트 구현하기
def XOR(x1, x2): 
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def FULL_ADDER(x1, x2, c_in):
    s1 = XOR(x1, x2)
    c1 = AND(x1, x2)
    s2 = XOR(s1, c_in)
    c2 = AND(s1, c_in)
    c3 = OR(c2, c1)
    return((s2, c3)) 
    
print(FULL_ADDER(0,0,0)) 
print(FULL_ADDER(0,0,1))
print(FULL_ADDER(0,1,0))
print(FULL_ADDER(0,1,1))
print(FULL_ADDER(1,0,0))
print(FULL_ADDER(1,0,1))
print(FULL_ADDER(1,1,0))
print(FULL_ADDER(1,1,1))