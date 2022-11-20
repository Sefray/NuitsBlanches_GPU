import argparse
import cv2
import json
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Video Maker',
        description='Create a video from a list of images readen from stdin')

    parser.add_argument('-o', '--output_filename',
                        type=str, default='output_video.avi')
    parser.add_argument('-f', '--fps', type=int, default=10)
    args = parser.parse_args()

    in_json = ""
    for line in sys.stdin:
        in_json += line.strip()

    in_dict = json.loads(in_json)

    img_frame_size = cv2.imread(list(in_dict.keys())[0]).shape[:2]
    frame_size = img_frame_size[1], img_frame_size[0]

    out = cv2.VideoWriter(args.output_filename, cv2.VideoWriter_fourcc(
        *'DIVX'), args.fps, frame_size)

    for file in in_dict:
        img = cv2.imread(file)
        for rect in in_dict[file]:
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(img)

    out.release()
