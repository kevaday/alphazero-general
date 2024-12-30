import matplotlib.pyplot as plt
import numpy as np
data = []
with open("elo/U3T/ELOS.csv",'r') as data_file:
	for line in data_file:
		data = line.split(",")

for i in range(len(data)):
	data[i] = float(data[i])

kernel_size = 7
kernel = np.ones(kernel_size) / kernel_size
data_convolved = np.convolve(data, kernel, mode='valid')

print(len(data))
plt.plot(data_convolved)
plt.show()
