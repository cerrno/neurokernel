'''
@author: Amol Kapoor
@version: 0.1
@date: 8-9-15

Connects to a socket and sends realtime input from arrays of 100 randomly generated numbers
'''


import socket
import time
import random
import json

ARRAYSIZE = 200*200;

def generateData(data_size):
    data = []

    for i in xrange(0, data_size):
        data.append(random.random()*100 - 50)

    return data

def main(data_size):
    #How often to send data (slightly over 30 frames per second)
    cycle_time = .03334


    #socket connection details
    host = 'localhost' 
    port = 50000 
    size = 4096
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s.connect((host,port)) 

    while 1: 
        start_time = time.time()

        data = generateData(data_size);

        data_str = json.dumps(data)

        print len(data_str)

        data_str = data_str + "_"

        s.send(data_str) 

        time.sleep(cycle_time)

if __name__ == "__main__":
    main(ARRAYSIZE)
