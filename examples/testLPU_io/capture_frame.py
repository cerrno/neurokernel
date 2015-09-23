# this is meant to be the cleanest prototype code for capturing images from
# a camera for use in the interface testing

# dependencies: video device connected via USB on port 0 (can modify below)
# libraries: opencv, (numpy), requisite dependencies and vanilla python dist

import cv2

import socket
import time
import random

try:
    import ujson as json
except ImportError:
    print "ERROR"
    import json

camera_port = 0

camera = cv2.VideoCapture(camera_port)

ARRAYSIZE = 640*480;

def get_image():
    retval, im = camera.read()
    return im

def main(data_size)
    #socket connection details
    host = 'localhost' 
    port = 50000 
    size = 4096
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s.connect((host,port)) 

    cycle_time = .034

    while 1: 
        start_time = current_milli_time()

        gray_image = cv2.cvtColor(get_image(), cv2.COLOR_BGR2GRAY))
            
        data = json.dumps(gray_image) + "_"

        s.send(data_array[count]) 

        time.sleep(max(0, cycle_time - start_time))

if __name__ == "__main__":
    main(ARRAYSIZE)
