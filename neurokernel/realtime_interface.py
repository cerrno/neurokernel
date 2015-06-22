#!/usr/bin/env python

"""
Real time sensor interface. Initializes N number of ports for use by a data field that maps to an array. 
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from neurokernel.base import PORT_DATA, PORT_CTRL
from neurokernel.core import Module

from neurokernel.plsel import PathLikeSelector
from neurokernel.tools.comm import get_random_port

class io_interface(Module):
    
    def __init__(self, num_ports, id, device):

        self.num_ports = num_ports
        self.cached_data = np.zeros(num_ports)
        self.selector_array = []

        #generate output ports
        for i in xrange(0, num_ports):
            selector_name = id + '/out/gpot/' + str(i)
            self.selector_array.append(selector_name)
            self.interface[selector_name, 'io', 'type'] = ['out', 'gpot']

        #generate input ports
        selector_name = id+'in/gpot/0'
        self.selector_array.append(selector_name)
        self.interface[selector_name, 'io', 'type'] = ['in', 'gpot']

        sel = ','.join(self.selector_array)

        count = PathLikeSelector.count_ports(sel)

        port_data = get_random_port()
        port_ctrl = get_random_port()

        super(io_interface, self).__init__(sel, sel, data_gpot=np.zeros(count, np.float64),
                                    columns=['interface', 'io', 'type'], port_data=port_data, 
                                    port_ctrl=port_ctrl, id=id, device=device)


    def get_data(array_data):
        if len(array_data) != len(self.num_ports):
            raise ValueError("Incoming array must have same length as number of interface ports")
        self.cached_data = array_data

    def send_data():
        # Output random graded potential data:
        out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
        self.pm['gpot'][out_gpot_ports] = self.cached_data

    @property
    def cached_data(self):
        return self.cached_data

    def run_step(self):
        super(io_interface, self).run_step()
        send_data()

        
