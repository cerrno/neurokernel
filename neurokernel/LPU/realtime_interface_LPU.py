#!/usr/bin/env python

"""
@author: Amol Kapoor
@version: 0.1
@date: 8-1-15

Real time interface for use with LPUs
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np

import pycuda.driver as cuda

from neurokernel.base import PORT_DATA, PORT_CTRL
from neurokernel.core import Module

from neurokernel.plsel import SelectorMethods
from neurokernel.tools.comm import get_random_port

from LPU import LPU

import h5py
import tables
import neurokernel.LPU.utils.simpleio as si

import sys

import threading
import socket
import time
import random

import os

try:
    import ujson as json
except ImportError:
    import json

from Queue import Queue

current_milli_time = lambda: int(round(time.time() * 1000))

class io_interface(LPU):

    #_input_data = []

    def input_server(self):
        host = '' 
        port = 60000
        backlog = 5 

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((host,port))
        s.listen(backlog)
        client, address = s.accept() 

        data_str = ""
        data_list = []
        data_buffer = ""

        
        self.log_info("Thread 2: Entering while loop")

        while 1: 
            start_time = current_milli_time()

            data_str = ""

            if len(data_buffer):
                data_str = data_buffer
                data_buffer = ""

            sent_data = client.recv(16384) 

            data_str = data_str + sent_data

            delim_sent_data = data_str.split('_', 1)

            data_size = int(delim_sent_data[0])

            data_str = delim_sent_data[1]

            self.log_info("Thread 2: Appending data list")

            while len(data_str) < data_size:
                self.log_info("Thread 2: Appending data list (2)")
                self.log_info("Thread 2: %d" % len(data_str))
                self.log_info("Thread 2: %d" % data_size)
                data_str = data_str + client.recv(16384)

            self.log_info("Thread 2: Out of loop")

            data_buffer = data_str[data_size:]

            self.log_info("Thread 2: Loaded buffer")

            image_data = data_str[0:data_size]

            self.log_info("Thread 2: Loaded image data")

            image = np.fromstring(image_data, dtype=np.uint8)

            image = image.astype(np.double)
            #image = np.unpackbits(image)

            self._input_data = image 

            self.log_info("Thread 2: %d" % (current_milli_time() - start_time))

    def __init__(self, dt, n_dict, s_dict, input_file, output_file, port_ctrl, port_data, port_time, device, id, debug, num_ports):

        self.num_ports = num_ports
        self._cached_data = np.zeros(num_ports)
        
        if not input_file:
            self._input_data = np.zeros(640*480, np.double)
        else:
            #Baked
            self.input_file = input_file
            self.index = 0
           
        super(io_interface, self).__init__(dt, n_dict, s_dict, input_file, output_file, device, port_ctrl, port_data, port_time, id, debug)

    def _read_realtime_input(self):
        print "||"
        print self.synapse_state

        self.synapse_state.set(self._cached_data)

    def process_data(self):
        '''
        Used to process the data before it is sent to neurons. Manipulate self._cached_data.

        Default randomly chooses total_synapse num of neurons from class attribute _input_data
        '''

        image = np.reshape(self._input_data, [1, 480, 640])

        if not os.path.isfile('imagedata'):
            si.write_array(image, 'imagedata', mode = 'w')
        else:
            si.write_array(image, 'imagedata', mode = 'a')
       
        self._cached_data = np.array(self._input_data[:len(self.synapse_state)])

    def get_data(self):
        if self.input_file:
            self._cached_data = self._input_data
            if self.index < len(self._input_data) - 1:
                self.index = self.index + 1
        else:
            self.process_data()
    '''
    def pre_run(self):
        super(io_interface, self).pre_run()
    '''
     
    def run_step(self):

        if self.first_step:
            self.log_info("HELLO");
            threading.Thread(target=self.input_server).start()

        self.get_data()
        super(io_interface, self).run_step()
         
    @property
    def cached_data(self):
        return self._cached_data
