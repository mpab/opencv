#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import time

# local modules
import sys
sys.path.append('../samples/python')

from video import create_capture
from common import clock, draw_str

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def get_cams():
    #cam = create_capture('synth:bg=monkey-race-wide.jpg:noise=0.05', None)

    cams = [
        create_capture('synth:bg=monkey-race-wide.jpg'),
        create_capture('synth:bg=monkey-race-clown.jpg'),
        create_capture('synth:bg=monkey-race-kids.jpg')
        ]

    return cams

def get_algos():
    algo_names = [
        "../data/haarcascades/haarcascade_frontalcatface.xml",
        "../data/haarcascades/haarcascade_frontalcatface_extended.xml",
        "../data/haarcascades/haarcascade_frontalface_alt.xml",
        "../data/haarcascades/haarcascade_frontalface_alt2.xml",
        "../data/haarcascades/haarcascade_frontalface_alt_tree.xml",
        "../data/haarcascades/haarcascade_frontalface_default.xml",
        "../data/haarcascades/haarcascade_profileface.xml",

        "../data/lbpcascades/lbpcascade_frontalcatface.xml",
        "../data/lbpcascades/lbpcascade_frontalface_improved.xml",
        "../data/lbpcascades/lbpcascade_frontalface.xml",
        "../data/lbpcascades/lbpcascade_profileface.xml",
        "../data/lbpcascades/lbpcascade_silverware.xml"
    ]

    algos = []
    for a in algo_names:
        algos.append(cv2.CascadeClassifier(a))

    return algo_names, algos

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

    video_src = 0
    args = dict(args)
    #cascade_fn = args.get('--cascade', "../data/haarcascades/haarcascade_frontalface_alt.xml")
    
    nested_fn  = args.get('--nested-cascade', "../data/haarcascades/haarcascade_eye.xml")
    
    algo_names, algos = get_algos()
    cams = get_cams()

    nested = cv2.CascadeClassifier(nested_fn)

    algo_idx = 0
    cam_idx = 0

    while True:
        cam = cams[cam_idx]

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, algos[algo_idx])

        algo_info = 'algorithm = {}:{}'.format(algo_idx, algo_names[algo_idx])

        cam_info = 'image = {}'.format(cam_idx)

        algo_idx += 1
        if (algo_idx >= len(algos)):
            algo_idx = 0

            cam_idx += 1
            if (cam_idx >= len(cams)):
                cam_idx = 0 


        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        draw_str(vis, (20, 40), algo_info)
        draw_str(vis, (20, 60), cam_info)

        cv2.imshow('face recognition', vis)

        time.sleep(1)

        if cv2.waitKey(50) == 27:
            break
    cv2.destroyAllWindows()
