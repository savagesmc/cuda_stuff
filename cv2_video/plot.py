import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster as sc

time = []
value = []

with open('output.csv') as csv_file:
   csv_reader = csv.reader(csv_file, delimiter=',')
   lineCount = 0
   for row in csv_reader:
      time.append(float(row[0]))
      value.append(float(row[1]))

print("data input")

time = np.reshape(time, (len(time), 1))
value = np.reshape(value, (len(value), 1))
centroids,_  = sc.vq.kmeans(value, 2)
idx,_ = sc.vq.vq(value, centroids)

print(len(value), len(idx))

for i in range(len(idx)):
   print("{} : {} : {}".format(i, value[i], idx[i]))

plt.plot(time, idx)
plt.show()
