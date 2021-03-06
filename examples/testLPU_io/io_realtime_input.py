'''
@author: Amol Kapoor
@version: 0.1
@date: 8-9-15

Connects to a socket and sends realtime input from arrays of 100 randomly generated numbers
'''


import socket
import time
import random

try:
    import ujson as json
except ImportError:
    print "ERROR"
    import json

ARRAYSIZE = 640*480;

current_milli_time = lambda: int(round(time.time() * 1000))

def main(data_size):
    #How often to send data (slightly over 30 frames per second)
    cycle_time = .034

    data_array = []

    for i in xrange(0, 3500):
        data = [round(random.random()*100 - 50) for _ in xrange(data_size/3)]
        data_array.append(json.dumps(data) + "_")
        print len(data_array)

    #socket connection details
    host = 'localhost' 
    port = 50000 
    size = 4096
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s.connect((host,port)) 

    print s

    count = 0;

    while 1: 
        start_time = current_milli_time()

        #print (len(data_array[count])*2)
        s.send(data_array[count]) 

        #print current_milli_time() - start_time

        time.sleep(max(0, cycle_time - start_time))
	
	count = count + 1;
	

if __name__ == "__main__":
    main(ARRAYSIZE)
