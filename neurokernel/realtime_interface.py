#!/usr/bin/env python

"""
Real time sensor interface. Initializes N number of ports for use by a data field that maps to an array. 
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from neurokernel.base import PORT_DATA, PORT_CTRL
from neurokernel.core import Module

from neurokernel.plsel import SelectorMethods
from neurokernel.tools.comm import get_random_port

class io_interface(Module):
    
    def __init__(self, num_ports, id, device, port_data, port_ctrl, port_time):

        self.num_ports = num_ports
        self._cached_data = np.zeros(num_ports)
        '''
        FOR TESTING
        '''
        #self._cached_data.fill(60)
        self.selector_array = []

        #generate output ports
        for i in xrange(0, num_ports):
            selector_name = '/' + id + '/out/gpot/' + str(i)
            self.selector_array.append(selector_name)

        selector_in_string = ','.join(self.selector_array)

        #generate input ports
        selector_out_string = '/' + id+'/in/gpot/0'
        self.selector_array.append(selector_out_string)

        selector_total_string = ','.join(self.selector_array)

        count = len(self.selector_array)

        super(io_interface, self).__init__(selector_total_string, selector_in_string, selector_out_string, sel_gpot = selector_total_string, sel_spike='', data_gpot=np.zeros(count, np.float64), data_spike = [], columns=['interface', 'io', 'type'], port_data=port_data, port_ctrl=port_ctrl, port_time=port_time, id=id, device=device)

        for selector_name in self.selector_array:
            self.interface[selector_name, 'io', 'type'] = ['out', 'gpot']
        
        self.interface[selector_out_string, 'io', 'type'] = ['in', 'gpot']

    def get_data(self, array_data):
        if len(array_data) != len(self.num_ports):
            raise ValueError("Incoming array must have same length as number of interface ports")
        self._cached_data = array_data

    def send_data(self):
        out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
        self.pm['gpot'][out_gpot_ports] = self._cached_data

    @property
    def cached_data(self):
        return self._cached_data

    def run_step(self):
        super(io_interface, self).run_step()
        self.send_data()

        
