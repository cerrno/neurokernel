#!/usr/bin/env python

"""
Real time sensor interface. Initializes N number of ports for use by a data field that maps to an array. 

Needs to be tested with multiple ports (640 * 480), and with async time with sockets
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from neurokernel.base import PORT_DATA, PORT_CTRL
from neurokernel.core import Module

from neurokernel.plsel import SelectorMethods
from neurokernel.tools.comm import get_random_port

import h5py
import tables
import neurokernel.LPU.utils.simpleio as si

import sys

import threading
import socket
import time

import LPU

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
            a = np.zeros(self.num_ports, np.float64)
            a.fill(-60)
            self._input_data.put(a)
            time.sleep(1)
            '''
            client, address = s.accept() 
            data = client.recv(size) 
            if data: 
                actual_data = float(data)
                a = np.zeros(self.num_ports, np.float64)
                a.fill(actual_data)
                self.__class__._input_data.append(a)
                client.send(data)
                self.log_info("Updated: %d" % actual_data)
            client.close()

            temp = " ".join(map(str,self.__class__._input_data))
            self.log_info("Thread 1: %s" % temp)

    def __init__(self, dt, n_dict, s_dict, input_file, output_file, port_ctrl, port_data, port_time, device, id, debug, num_ports):

        self.num_ports = num_ports
        self._cached_data = np.zeros(num_ports)
        
        #Baked
        if !input_file:
            init = np.zeros(num_ports, np.float64)
            self.__class__._input_data.append(init)
        else:
            self.input_file = input_file
            self.index = 0
           
        super(io_interface, self).__init__(self, dt, n_dict, s_dict, input_file, output_file, port_ctrl, port_data, port_time, device, id, debug)

    def _read_realtime_input(self):
        '''
        Passes cahced info to gpu
        '''
        print self._cached_data

    def get_data(self):
        if self.input_file:
            self._cached_data = self.__class__._input_data
            if self.index < len(self.__class__._input_data) - 1:
                self.index = self.index + 1
        else:
            self._cached_data = self.__class__._input_data[len(self.__class__._input_data) - 1]

    def pre_run(self):
        super(io_interface, self).pre_run()
        threading.Thread(target=self.input_server).start()
     
    def run_step(self):
        self.get_data()
        self.process_data()
        super(io_interface, self).run_step()

    def process_data(self):
        '''
        Used to process the data before it is sent to neurons. Manipulate self._cached_data.
        '''
        print 'stuff'
         
    @property
    def cached_data(self):
        return self._cached_data
