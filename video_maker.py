import cv2
import numpy as np
import glob
import sys

import json

if __name__ == "__main__":
    frameSize = (360, 640)

    out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

    in_json = ""
    for line in sys.stdin:
        in_json += line.strip()

    in_dict = json.loads(in_json)

    for file in in_dict:
        img = cv2.imread(file)
        for rect in in_dict[file]:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        out.write(img)

    # display in_dict on stdout as a string
    # print(json.dumps(in_dict))

    # for filename in glob.glob('D:/images/*.jpg'):
    #     img = cv2.imread(filename)

    out.release()
