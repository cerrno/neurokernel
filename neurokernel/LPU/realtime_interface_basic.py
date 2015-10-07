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

class io_interface(Module):

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

    def __init__(self, num_ports, id, device, port_data, port_ctrl, port_time, filename=None):
        self.num_ports = num_ports
        self._cached_data = np.zeros(num_ports)
        
        self.filename = filename

        #Baked
        if filename:
            self.__class__._input_data = si.read_array('./data/simple_input.h5')
            self.index = 0
        #Realtime
        else:
            init = np.zeros(num_ports, np.float64)
            self.__class__._input_data.append(init)
           
        self.selector_array = []

        #generate output ports
        for i in range(0, num_ports):
            selector_name = '/' + id + '/out/gpot/' + str(i)
            self.selector_array.append(selector_name)

        selector_in_string = ','.join(self.selector_array)

        selector_out_string = '/' + id+'/in/gpot/0'
        self.selector_array.append(selector_out_string)

        selector_total_string = ','.join(self.selector_array)

        count = len(self.selector_array)

        super(io_interface, self).__init__(selector_total_string, selector_in_string, selector_out_string, sel_gpot = selector_total_string, sel_spike='', data_gpot=np.zeros(count, np.float64), data_spike = [], columns=['interface', 'io', 'type'], port_data=port_data, port_ctrl=port_ctrl, port_time=port_time, id=id, device=device)

        for selector_name in self.selector_array:
            self.interface[selector_name, 'io', 'type'] = ['out', 'gpot']

        self.interface[selector_out_string, 'io', 'type'] = ['in', 'gpot']

    def get_data(self):
        if self.filename:
            self._cached_data = self.__class__._input_data
            if self.index < len(self.__class__._input_data) - 1:
                self.index = self.index + 1
        else:
            self._cached_data = self.__class__._input_data[len(self.__class__._input_data) - 1]

    def send_data(self):
        out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()

        self.pm['gpot'][out_gpot_ports] = \
                    self._cached_data

    def pre_run(self):
        #set up socket handling
        threading.Thread(target=self.input_server).start()
     
    def run_step(self):
        super(io_interface, self).run_step()
        self.get_data()
        self.send_data()


        temp = " ".join(map(str,self.__class__._input_data))
        self.log_info("Thread 2: %s" % temp)
     
    @property
    def cached_data(self):
        return self._cached_data
