import cv2, argparse, json, sys, csv, datetime
import subprocess as sp
import numpy as np
import scipy, scipy.signal
import scipy.cluster as sc
import matplotlib.pyplot as plt


''' Give a json from ffprobe command line

@vid_file_path : The absolute (full) path of the video file, string.
'''
def probe(vid_file_path):
   if type(vid_file_path) != str:
      raise Exception('Gvie ffprobe a full file path of the video')
      return

   command = ["ffprobe",
      "-loglevel",  "quiet",
      "-print_format", "json",
      "-show_format",
      "-show_streams",
      vid_file_path
      ]

   pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
   out, err = pipe.communicate()
   return json.loads(out.decode('utf-8'))


''' Video's duration in seconds, return a float number
'''
def duration(vid_file_path):
   _json = probe(vid_file_path)

   if 'format' in _json:
      if 'duration' in _json['format']:
         return float(_json['format']['duration'])

   if 'streams' in _json:
      # commonly stream 0 is the video
      for s in _json['streams']:
         if 'duration' in s:
            return float(s['duration'])

   # if everything didn't happen,
   # we got here because no single 'return' in the above happen.
   raise Exception('I found no duration')
   #return None

def doFilter(sig, N=256, Fs=0.1, Fc=30):
   h = scipy.signal.firwin(numtaps=N, cutoff=Fc, nyq=Fs/2)
   y = scipy.signal.lfilter(h, 1.0, sig)
   return y

def inside(r, q):
   rx, ry, rw, rh = r
   qx, qy, qw, qh = q
   return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
   for x, y, w, h in rects:
      # the HOG detector returns slightly larger rectangles than the real objects.
      # so we slightly shrink the rectangles to get a nicer output.
      pad_w, pad_h = int(0.15*w), int(0.05*h)
      cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (150, 150, 220), thickness)

class Finder:
   def __init__(self, fname, ofname):
      self.cap = cv2.VideoCapture(args.filename)
      self.ofname = ofname
      if self.ofname:
         self.out = cv2.VideoWriter('output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10,        # frames per sec
                          (853,480)) # frame resolution
      self.hog = cv2.HOGDescriptor()
      self.hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
      self.fps = self.cap.get(cv2.CAP_PROP_FPS)
      self.vidLength = duration(args.filename)
      print("{} : {}".format(self.vidLength, self.fps))

   def __del__(self):
      self.cap.release()

   def analyze(self, interval=30, ofname="", debug=False):
      finds = []
      est_tot_frames = self.vidLength * self.fps  # Sets an upper bound # of frames in video clip
      n = interval # Desired interval of frames to include
      desired_frames = n * np.arange(est_tot_frames/n)
      if ofname:
         ofile = open(ofname, "wb")
      last = 0
      for i in desired_frames:
         self.cap.set(1,i-1)
         ret, img = self.cap.read(1)
         if ret:
            frameId = self.cap.get(1) # The 0th frame is often a throw-away
            if debug:
               print("frame : {}".format(frameId))
            found, w = self.hog.detectMultiScale(img, hitThreshold=0.2, winStride=(8,8), padding=(32,32), scale=1.05)
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and inside(r, q):
                        break
                else:
                    found_filtered.append(r)
            finds.append((i, found, w, found_filtered))
            if self.ofname or debug:
               draw_detections(img, found)
               draw_detections(img, found_filtered, 3)
               resize = cv2.resize(img, (853, 480), interpolation=cv2.INTER_AREA)
            if self.ofname:
               self.out.write(resize)
            if debug:
               print('%d (%d) found' % (len(found_filtered), len(found)))
               cv2.imshow('frame', resize)
            if cv2.waitKey(1) & 0xff == ord('q'):
               break;
         if debug or (i - last) > 3000:
            print(i)
            last = i
      return finds

def doKmeans(time, values, debug=False):
   centroids,_ = sc.vq.kmeans(values, 2)
   idx,_ = sc.vq.vq(values, centroids)
   idx = ((idx - 0.5) * -2. + 1.) / 2.
   if debug:
      print(len(value), len(idx))
      for i in range(len(idx)):
         debug("{} : {} : {}".format(i, value[i], idx[i]))
      plt.plot(time, idx)
      plt.show()
   return idx

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-n", "--filename", help="name of video file to process", default="video.mp4")
   parser.add_argument("-i", "--interval", help="frame skip interval", type=int, default=30)
   parser.add_argument("-c", "--csvoutput", help="write analysis to csv", default="")
   parser.add_argument("--csvinput", help="read analysis from csv", default="")
   parser.add_argument("-o", "--output", help="write bounding box to video avi file", default="")
   parser.add_argument("-Fs", help="sample frequency (hz)", type=float, default = 30)
   parser.add_argument("-Fc", help="low pass filter center frequency (hz)", type=float, default = 0.01)
   parser.add_argument("-N", help="low pass filter number of taps", type=int, default = 256)
   parser.add_argument("-d", "--debug", help="debug", action='store_true')
   parser.add_argument("--printOptions", action='store_true')
   args = parser.parse_args()

   if args.printOptions:
      for arg in vars(args):
         print("{:20} = {}".format(arg, getattr(args, arg)))
      sys.exit(0)

   if not args.csvinput:
      finder = Finder(args.filename, args.output)
      stats = finder.analyze(interval=args.interval, debug=args.debug)
      cv2.destroyAllWindows()
      times = np.array([s[0]/finder.fps for s in stats])
      num = doFilter(np.array([len(s[1]) for s in stats]), args.N, args.Fs, args.Fc)
      classified = doKmeans(times, num, debug=args.debug)

   else:
      times = []
      num = []
      classified = []

      with open(args.csvinput) as csv_file:
         csv_reader = csv.reader(csv_file, delimiter=',')
         lineCount = 0
         for row in csv_reader:
            times.append(float(row[0]))
            num.append(float(row[1]))
            classified.append(float(row[2]))
         print("data input")

      times = np.array(times)
      num = np.array(num)
      classified = np.array(classified)
      for i in range(len(classified)):
         print("{} : {} : {} : {}".format(i, times[i], num[i], classified[i]))

   if args.csvoutput:
      with open(args.csvoutput, "w") as ofile:
         for i in range(len(times)):
            ofile.write("{},{},{}\n".format(times[i], num[i], classified[i]))

   starts = []
   ends = []
   dur = []
   prev=0
   for i in range(len(times)):
      cur = classified[i]
      if cur != prev:
         if cur == 1:
            starts.append(times[i])
         else:
            ends.append(times[i])
            if len(starts) != len(ends):
              starts.append(times[i])
         prev = cur

   for i in range(min(len(starts), len(ends))):
      dur.append(ends[i] - starts[i])

   starts = np.array(starts)
   ends = np.array(ends)
   dur = np.array(dur)

   def getTime(x):
      hour = int(x/3600)
      x -= 3600*hour
      minute = int(x/60)
      x -= 60*minute
      second = int(x)
      x -= second
      return datetime.time(hour=hour, minute=minute, second=second, microsecond=int(x*1e6))

   sermonIdx = np.argmax(dur)
   startTime = getTime(starts[sermonIdx]-args.N/2)
   endTime = getTime(ends[sermonIdx]-args.N/2)
   print("Sermon Start:  {}  {}  {}".format(startTime, endTime, getTime(dur[sermonIdx])))

   plt.plot(times, num)
   plt.plot(times, classified*max(num))
   plt.show()
