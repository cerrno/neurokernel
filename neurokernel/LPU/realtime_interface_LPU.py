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
import json

from Queue import Queue

class io_interface(LPU):

    _input_data = []

    def input_server(self):
        host = '' 
        port = 50000 
        backlog = 5 
        size = 4096
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((host,port)) 
        s.listen(backlog) 

        client, address = s.accept() 

        data_str = ""
        data_buffer = ""

        while 1: 
            self.log_info("Thread 0: starting")

            if len(data_buffer):
                data_str = data_buffer

            while 1:
                data_str = data_str + client.recv(size)
                if '_' in data_str:
                    break

            data_info = data_str.split('_')

            data = data_info[0]

            data_buffer = data_info[1]

            self.log_info("Thread 2: %d" % len(data))

            if data: 
                actual_data = json.loads(data)

                a = np.array(actual_data)

                self.__class__._input_data.append(a)

                self.log_info("Thread 3: loaded")

    def __init__(self, dt, n_dict, s_dict, input_file, output_file, port_ctrl, port_data, port_time, device, id, debug, num_ports):

        self.num_ports = num_ports
        self._cached_data = np.zeros(num_ports)
        
        #Baked
        if not input_file:
            init = np.zeros(num_ports, np.float64)
            self.__class__._input_data.append(init)
        else:
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

        self._cached_data = np.array(self.__class__._input_data[len(self.__class__._input_data) - 1][:len(self.synapse_state)])

    def get_data(self):
        if self.input_file:
            self._cached_data = self.__class__._input_data
            if self.index < len(self.__class__._input_data) - 1:
                self.index = self.index + 1
        else:
            self.process_data()

    def pre_run(self):
        super(io_interface, self).pre_run()
        threading.Thread(target=self.input_server).start()
     
    def run_step(self):
        self.get_data()
        super(io_interface, self).run_step()
         
    @property
    def cached_data(self):
        return self._cached_data
