#!/usr/bin/python3
import matplotlib.pyplot as plt
import csv, math
import numpy as np
from optparse import OptionParser

def doPlot(fname, fftshift, N):
   mags = []
   samples = []
   with open(fname, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      for row in spamreader:
         i = float(row[0])
         q = float(row[1])
         sample = complex(float(row[0]), float(row[1]))
         samples.append(sample)

   samples = np.array(samples)

   if (fftshift):
      samples = np.fft.fftshift(samples)

   plt.subplot(3,1,1)
   plt.plot(samples.real)
   plt.subplot(3,1,2)
   plt.plot(samples.imag)
   plt.subplot(3,1,3)
   plt.plot(abs(samples))
   plt.show()

if __name__ == "__main__":

   parser = OptionParser()

   parser.add_option("-f", "--file", dest="filename",
                     help="input FILE", metavar="FILE")

   parser.add_option("-F", "--fftshift",
                     action="store_true", dest="fftshift", default=False,
                     help="perform fft shift of data before plotting")

   parser.add_option("-N", "--sampleRate", dest="sampleRate",
                     default=1.,  help="sample rate of data")

   (options, args) = parser.parse_args()

   doPlot(options.filename, options.fftshift, options.sampleRate)
