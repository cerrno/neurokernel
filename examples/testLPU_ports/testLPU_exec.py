#@author: Amol Kapoor
#@date: 3-13-15
#Executor of simple LPU defined in data

from neurokernel.core import Manager
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port
import neurokernel.base as base

from neurokernel.pattern import Pattern

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
#logger = base.setup_logger(file_name='neurokernel.log', screen=False)
port_data = get_random_port()
port_ctrl = get_random_port()
port_time = get_random_port()

(n_dict, s_dict) = LPU.lpu_parser('./data/simple_lpu_0.gexf.gz')

lpu_0 = LPU(dt, n_dict, s_dict, input_file='./data/simple_input.h5', output_file='simple_output_0.h5', port_ctrl=port_ctrl, port_data=port_data, port_time=port_time, device=0, id='lpu_0', debug=False)

(n_dict, s_dict) = LPU.lpu_parser('./data/simple_lpu_1.gexf.gz')

lpu_1 = LPU(dt, n_dict, s_dict, input_file=None, output_file='simple_output_1.h5', port_ctrl=port_ctrl, port_data=port_data, port_time=port_time, device=1, id='lpu_1', debug=False)

#____________________________________________________________

out_ports_gpot_0 = '/lpu_0/out/gpot/1'
in_ports_gpot_1 = '/lpu_1/in/gpot/0'

pat = Pattern(out_ports_gpot_0, in_ports_gpot_1)

pat.interface[out_ports_gpot_0] = [0, 'out', 'gpot']
pat.interface[in_ports_gpot_1] = [1, 'in', 'gpot']

pat[out_ports_gpot_0, in_ports_gpot_1] = 1

#_________________________________________________
man = Manager(port_data, port_ctrl, port_time)

man.add_brok()

man.add_mod(lpu_0)
man.add_mod(lpu_1)

man.connect(lpu_0, lpu_1, pat, 0, 1)

man.start(steps=Nt)

man.stop()

