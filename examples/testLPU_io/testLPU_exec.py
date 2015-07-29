#@author: Amol Kapoor
#@date: 3-13-15
#Executor of simple LPU defined in data

from neurokernel.core import Manager
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port
import neurokernel.base as base

from neurokernel.LPU.realtime_interface_LPU import io_interface

from neurokernel.pattern import Pattern

import sys

sys.stdout.flush()

def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":
        indent[0] += 2
        print "-" * indent[0] + "> call function", frame.f_code.co_name
    elif event == "return":
        print "<" + "-" * indent[0], "exit function", frame.f_code.co_name
        indent[0] -= 2
    return tracefunc

import sys
#sys.settrace(tracefunc)

logger = base.setup_logger(file_name='neurokernel.log', screen=False)

dt = 1e-4
dur = 1.0
Nt = int(dur/dt)

port_data = get_random_port()
port_ctrl = get_random_port()
port_time = get_random_port()

#init the realtime interface

num_ports = 100
(n_dict, s_dict) = LPU.lpu_parser('./data/simple_lpu_1.gexf.gz')

lpu_1 = io_interface(dt, n_dict, s_dict, input_file=None, output_file='simple_output_1.h5', port_ctrl=port_ctrl, port_data=port_data, port_time=port_time, device=1, id='lpu_1', debug=False, num_ports=num_ports)

man = Manager(port_data, port_ctrl, port_time)

man.add_brok()

man.add_mod(interface)
man.add_mod(lpu_1)

man.connect(interface, lpu_1, pat, 0, 1)

man.start(steps=Nt)

man.stop()

