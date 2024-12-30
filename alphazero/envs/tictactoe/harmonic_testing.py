import numpy as np


def harmonic(n):
	a = 0
	for i in range(0, n):
		a += 1/(1+i)
	return a

GG = (1/harmonic(10)) / (1+np.arange(0, 10))
print(np.random.choice(np.arange(0, 10), p=GG, size=10))