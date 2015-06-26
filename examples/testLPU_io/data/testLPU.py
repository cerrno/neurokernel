#An implementation of a 2 neuron lpu that instantiates a Leaky and a Morrislecar 
#into a 1 edge/2node graph. Outputs to a .gefx file. Take from py notebooks. 

#matrix support
import numpy as np
np.set_printoptions(threshold = 'nan')
#graph support
import networkx as nx

#input support
import h5py

def create_lpu_0():
    #sets up neurons/networks
    G = nx.DiGraph()

    #sets up node connections in graph
    G.add_nodes_from([0,1])

    G.node[0] = {
            'name': 'port_in_gpot_0',
            'model': 'port_in_gpot',
            'selector': '/lpu_1/in/gpot/0', 
            'spiking': False,
            'public': True,
            'extern': False
            }

    #MorrisLecar updated
    G.node[1] = {
        'model': 'HodgkinHuxley_RK4',
        'name': 'neuron_1',
        'extern': False, 
        'public': True, 
        'spiking': False, 
        'selector':'/lpu_1/out/gpot/0', 
        'initV': -65.0,
        'initn': 0.0003,
        'initm': 0.0011,
        'inith': 0.9998,
        'C_m': 1,
        'V_Na': 55,
        'V_K': -77,
        'V_l': -54.4,
        'g_Na': 120,
        'g_K': 36,
        'g_l': 0.3
    }

        
    #From input port to output
    G.add_edge(0, 1, type='directed', attr_dict={
        'name': G.node[0]['name']+'-'+G.node[1]['name'],
        'model'       : 'power_gpot_gpot_sig',
        'class'       : 3,
        'slope'       : 0.8,
        'threshold'   : -45,
        'power'       : 10.0,
        'saturation'  : 30.0,
        'delay'       : 1.0,
        'reverse'     : 0,
        'conductance' : True})



    nx.write_gexf(G, 'simple_lpu_1.gexf.gz')


#sets up input file

#time step
dt = 1e-4
#duration
dur = 1.0
#number of datapoints
Nt = int(dur/dt)

#start and stop of input
start = 0.3
stop = 0.6

#the current input
I_max = 60
t = np.arange(0, dt*Nt, dt)
I = np.zeros((Nt, 1), dtype=np.double)

#inputs current at points indicated
I[np.logical_and(t > start, t < stop)] = I_max

with h5py.File('simple_input.h5', 'w') as f: 
    f.create_dataset('array', (Nt, 1), dtype = np.double, data = I)

create_lpu_0()
