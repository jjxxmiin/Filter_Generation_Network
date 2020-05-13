"""Mnist LRP Tutorial : http://heatmapping.org/tutorial/"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def show_digit(X):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(X, interpolation='nearest', cmap='gray')
    plt.show()


def show_heatmap(R):
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.show()


input = np.loadtxt('data/X.txt')
label = np.loadtxt('data/T.txt')

print(f"INPUT SHAPE : {input.shape} \t LABEL SHAPE : {label.shape}")

weight = [np.loadtxt('params/l%d-W.txt'%l) for l in range(1, 4)]
bias = [np.loadtxt('params/l%d-B.txt'%l) for l in range(1, 4)]

for i, (w, b) in enumerate(zip(weight, bias)):
    print(f"{i}-th weight : {w.shape}")
    print(f"{i}-th bias   : {b.shape}")

L = len(weight)

a = [input] + [None] * L

# activation function relu
# input, 1-layer output, 2-layer output, 3-layer output
for i in range(L):
    a[i + 1] = np.maximum(0, a[i] @ weight[i] + bias[i])

for i in range(L):
    # show_digit(input[i].reshape(28,28))
    p = a[L][i]
    print(f"predict : {np.argmax(p)} \t score : {p[np.argmax(p)]}")

R = [None]*L + [a[L]*(label[:,None]==np.arange(10))]

"""
R[k + 1] = sum(a[k] @ rho(w[j,k]) / (sum(a[j] @ rho(w[j, k])) + epsilon)) * R[k]

special case
upper  : LRP 0
middle : LRP epsilon
lower  : LRP gamma
"""


def rho(w, l):
    return w + [None, 0.1, 0.0, 0.0][l] * np.maximum(0, w)


def incr(z, l):
    return z + [None, 0.0, 0.1, 0.0][l] * (z**2).mean()**.5+1e-9


for l in range(1, L)[::-1]:
    w = rho(weight[l], l)
    b = rho(bias[l], l)

    z = incr(a[l].dot(w) + b, l)  # step 1
    s = R[l + 1] / z  # step 2
    c = s.dot(w.T)  # step 3
    R[l] = a[l] * c  # step 4


w = weight[0]
wp = np.maximum(0, w)
wm = np.minimum(0, w)
lb = a[0]*0-1
hb = a[0]*0+1

z = a[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1
s = R[1]/z                                        # step 2
c, cp, cm = s.dot(w.T), s.dot(wp.T), s.dot(wm.T)  # step 3
R[0] = a[0]*c-lb*cp-hb*cm                         # step 4

show_digit(input.reshape(1, 12, 28, 28).transpose(0, 2, 1, 3).reshape(28, 12*28))
show_heatmap(R[0].reshape(1, 12, 28, 28).transpose(0, 2, 1, 3).reshape(28, 12*28))
