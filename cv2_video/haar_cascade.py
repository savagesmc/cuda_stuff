import cv2, argparse
import numpy as np

#################### Setting up parameters ################

#OpenCV is notorious for not being able to good to
# predict how many frames are in a video. The point here is just to
# populate the "desired_frames" list for all the individual frames
# you'd like to capture.

def process(cap, out):
   fps = cap.get(cv2.CAP_PROP_FPS)
   est_video_length_minutes = 120         # Round up if not sure.
   est_tot_frames = est_video_length_minutes * 60 * fps  # Sets an upper bound # of frames in video clip
   n = 5                             # Desired interval of frames to include
   desired_frames = n * np.arange(1200, est_tot_frames)
   body_cascade = cv2.CascadeClassifier('haar/haarcascade_upperbody.xml')
   for i in desired_frames:
      cap.set(1,i-1)
      ret, img = cap.read(1)
      if ret:
         frameId = cap.get(1) # The 0th frame is often a throw-away
         print("frame : {}".format(frameId))
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=3, minSize=(60,60))
         for (x,y,w,h) in bodies:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            print("body")
         resize = cv2.resize(img, (853, 480), interpolation=cv2.INTER_AREA)
         cv2.imshow('frame', resize)
         out.write(resize)
      if cv2.waitKey(1) & 0xff == ord('q'):
         break;

if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("-n", "--filename", help="name of video file to process", default="video.mp4")
   args = parser.parse_args()
   cap = cv2.VideoCapture(args.filename)
   out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (853,480))
   if cap:
      process(cap, out)
      cap.release()
   cv2.destroyAllWindows()
