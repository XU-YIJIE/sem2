#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys, getopt
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets



def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:  #if no face or eyes in aimed region, return null list
        return []
    rects[:,2:] += rects[:,:2]
    return rects

class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets['cube'])
        _ret, self.frame = self.cam.read()
        
        args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
        try:
            video_src = video_src[0]
        except:
            video_src = 0
        args = dict(args)
        cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
        cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
        
        rectang = np.array([0, 0, 0, 0])
        
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        self.rects = detect(gray, cascade)
        if  len(self.rects) :
            rectang = np.array([self.rects[0][0], self.rects[0][1], self.rects[0][2], self.rects[0][3]])
        
        cv.namedWindow('camshift')
#        cv.setMouseCallback('camshift', self.onmouse)

#        self.selection = None
        self.selection = (rectang[0],rectang[1],rectang[2],rectang[3])
        self.drag_start = None
        self.show_backproj = False
#        self.track_window = None
        self.track_window = (rectang[0],rectang[1],rectang[2]-rectang[0],rectang[3]-rectang[1])

        
        
        


        
#        if event == cv.EVENT_LBUTTONDOWN:
#            self.drag_start = (x, y)
#            self.track_window = None
#        if self.drag_start:
#            xmin = min(x, self.drag_start[0])
#            ymin = min(y, self.drag_start[1])
#            xmax = max(x, self.drag_start[0])
#            ymax = max(y, self.drag_start[1])
#            self.selection = (xmin, ymin, xmax, ymax)
#        if event == cv.EVENT_LBUTTONUP:
#            self.drag_start = None
#            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def run(self):

        while True:

            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            
            
            
            
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            hand = []
            
            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                
                prob[self.rects[0][1]:self.rects[0][3],self.rects[0][0]:self.rects[0][2]] = 0
                    
                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)
                
                xh1,yh1,xh2,yh2 = self.track_window
                hand = prob[yh1:yh1+224,xh1:xh1+224]

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)
#                    V=vis
#                    print(self.track_window)
#                    break
                except:
                    print(track_box)

            cv.imshow('camshift', vis)
            
            
            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
                
            if ch == 32:
                hand2 = cv.resize(hand,(16,16),interpolation=cv.INTER_CUBIC)
                np.savetxt('1.csv',hand)
                cv.imwrite('1.png',hand)
                cv.imwrite('2.png',hand2)
                print(hand)
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    print(__doc__)
    App(video_src).run()
