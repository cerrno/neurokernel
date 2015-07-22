#@author: Amol Kapoor
#@date: 3-13-15
#Executor of simple LPU defined in data

from neurokernel.core import Manager
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port
import neurokernel.base as base

from neurokernel.realtime_interface import io_interface

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
id = 'interface_0'

interface = io_interface(num_ports, id, 0, port_data, port_ctrl, port_time) #'./data/simple_input.h5')

(n_dict, s_dict) = LPU.lpu_parser('./data/simple_lpu_1.gexf.gz')

lpu_1 = LPU(dt, n_dict, s_dict, input_file=None, output_file='simple_output_1.h5', port_ctrl=port_ctrl, port_data=port_data, port_time=port_time, device=1, id='lpu_1', debug=False)

#____________________________________________________________

out_port_gpot = []
in_port_gpot = []

for i in range(num_ports):
    out_port_gpot.append( '/' + id + '/out/gpot/' + str(i))
    in_port_gpot.append('/lpu_1/in/gpot/' + str(i*2))

out_ports_gpot = ','.join(out_port_gpot)
in_ports_gpot = ','.join(in_port_gpot)

pat = Pattern(out_ports_gpot, in_ports_gpot)

for i in range(num_ports):
    out_port_gpot = '/' + id + '/out/gpot/' + str(i)
    in_port_gpot = '/lpu_1/in/gpot/' + str(i*2)

    pat.interface[out_port_gpot] = [0, 'out', 'gpot']
    pat.interface[in_port_gpot] = [1, 'in', 'gpot']

    pat[out_port_gpot, in_port_gpot] = 1

#_________________________________________________
man = Manager(port_data, port_ctrl, port_time)

man.add_brok()

man.add_mod(interface)
man.add_mod(lpu_1)

man.connect(interface, lpu_1, pat, 0, 1)

man.start(steps=Nt)

man.stop()

