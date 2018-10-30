import csv
import matplotlib.pyplot as plt

time = []
value = []


with open('output.csv') as csv_file:
   csv_reader = csv.reader(csv_file, delimiter=',')
   lineCount = 0
   for row in csv_reader:
      time.append(row[0])
      value.append(row[1])

plt.plot(time, value)
plt.show()
