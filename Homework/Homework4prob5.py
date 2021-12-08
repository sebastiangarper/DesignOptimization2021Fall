## Calling modules

import numpy as np
import matplotlib as plt
import math

def fun(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    return x1**2+x2**2+x3**2


def Dfdd(x):
    return 2 * x[0]


def Dfds(x):
    return np.array([2 * x[1], 2 * x[2]])


def Dhds(x):
    return np.array([[.4 * x[1], 0.08 * x[2]], [1, -1]])


def Dhdd(x):
    return np.array([[x[0] / 2], [1]])


def Dfdd2(x):
    return Dfdd(x) - np.matmul(np.matmul(Dfds(x), np.linalg.inv(Dhds(x))), Dhdd(x))


def xeval(x, a, dfdd):
    d1 = (x[0] - a * dfdd)[0]
    s1 = x[1:3] + a * np.transpose(np.matmul(np.matmul(np.linalg.inv(Dhds(x)), Dhdd(x)), np.transpose([Dfdd2(x)])))[0]
    return np.append(d1, s1)


def ls(dfdd, x):
    a = 1
    b = .5
    t = .3
    while fun(xeval(x, a, dfdd)) > (fun(x) - a * t * dfdd ** 2):
        a = b * a
    return a


def solve(x):
    while np.linalg.norm(np.array([[x[0] ** 2 / 4 + x[1] ** 2 / 5 + x[2] ** 2 / 25 - 1], [x[0] + x[1] - x[2]]])) > 10 ** (-3):
        dhds = Dhds(x)
        skj1 = np.transpose(np.transpose([x[1:3]]) - np.matmul(np.linalg.inv(dhds), np.array(
            [[x[0] ** 2 / 4 + x[1] ** 2 / 5 + x[2] ** 2 / 25 - 1], [x[0] + x[1] - x[2]]])))
        x = np.append(x[0:1], np.transpose(skj1[0]))
    return x


x1 = 0
x3 = 1 / 12 * ((600 - 170 * (x1 ** 2)) ** (1 / 2) + 10 * x1)
x2 = x3 - x1

x0 = np.array([x1, x2, x3])
xfinal = [x0]
err = []

while np.linalg.norm(Dfdd2(xfinal[-1])) > 10 ** (-3):
    x = xfinal[-1]
    dfdd = Dfdd2(x)
    err.append(math.log(np.linalg.norm(dfdd)))
    a = ls(dfdd, x)
    dk = x[0] - a * dfdd
    sk0 = x[1:3] + a * np.transpose(np.matmul(np.matmul(np.linalg.inv(Dhds(x)), Dhdd(x)), np.transpose(dfdd)))
    xk0 = np.append(dk, sk0)
    x = solve(xk0)
    xfinal.append(x)

print('Solution ' + str(xfinal[-1]))
print('Hello World')