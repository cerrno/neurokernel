#!/usr/bin/env python

"""
Real time sensor interface. Initializes N number of ports for use by a data field that maps to an array. 

Needs to be tested with multiple ports (640 * 480), and with async time with sockets
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

from Queue import Queue

class io_interface(LPU):

    _input_data = []

    def input_server(self):
        host = '' 
        port = 50000 
        backlog = 5 
        size = 1024 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((host,port)) 
        s.listen(backlog) 
               
        while 1: 
            '''
            actual code, commented for testing:
            actual_data = json.loads(data)
            a = np.array(actual_data)
            '''
            client, address = s.accept() 
            data = client.recv(size) 
            if data: 
                actual_data = float(data)
                a = np.zeros(self.num_ports, np.float64)
                a.fill(actual_data)
                self.__class__._input_data.append(a)

                self.log_info("Updated: %d" % actual_data)
            client.close()

            temp = " ".join(map(str,self.__class__._input_data))
            self.log_info("Thread 1: %s" % temp)

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
        '''
        Passes cahced info to gpu
        '''
        print "||"
        print len(self.synapse_state)
        print self.synapse_state
        print self.synapse_state.dtype
        print "|"
        print self.total_synapses
        print "|"
        print self._cached_data
        print self._cached_data.shape
        print self._cached_data.dtype
        print "||"

        '''
        cuda.memcpy_htod(
            int(int(self.synapse_state.gpudata) +
            len(self.synapse_state)*self.synapse_state.dtype.itemsize),
            self._cached_data)
        '''

        self.synapse_state.set(self._cached_data)

    def process_data(self):
        '''
        Used to process the data before it is sent to neurons. Manipulate self._cached_data.

        Default randomly chooses total_synapse num of neurons from class attribute _input_data
        '''

        self._cached_data = np.zeros(len(self.synapse_state), np.float64)

        for i in xrange(0, len(self.synapse_state)):
            self._cached_data[i] = random.choice(self.__class__._input_data[len(self.__class__._input_data) - 1])

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
        self.process_data()
        super(io_interface, self).run_step()
         
    @property
    def cached_data(self):
        return self._cached_data
