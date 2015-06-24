#@author: Amol Kapoor
#@date: 3-13-15
#Executor of simple LPU defined in data

from neurokernel.core import Manager
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.comm import get_random_port
import neurokernel.base as base

dt = 1e-4
dur = 1.0
Nt = int(dur/dt)

port_data = get_random_port()
port_ctrl = get_random_port()

(n_dict, s_dict) = LPU.lpu_parser('./data/simple_lpu.gexf.gz')

lpu = LPU(dt, n_dict, s_dict, input_file='./data/simple_input.h5', output_file='simple_output.h5', port_ctrl=port_ctrl, port_data=port_data, device=0, id='simple', debug=False)

print n_dict
print
print s_dict

man = Manager(port_data, port_ctrl)

man.add_brok()

man.add_mod(lpu)

man.start(steps=Nt)

print "HELLO"

man.stop()
