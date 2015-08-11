#An implementation of a 2 neuron lpu that instantiates a Leaky and a Morrislecar 
#into a 1 edge/2node graph. Outputs to a .gefx file. Take from py notebooks. 

#matrix support
import numpy as np
np.set_printoptions(threshold = 'nan')
#graph support
import networkx as nx

#input support
import h5py

def create_lpu_0(neu_num):

    #sets up neurons/networks
    G = nx.DiGraph()

    #sets up node connections in graph
    G.add_nodes_from(range(neu_num))


    for i in xrange(0, neu_num):

        #MorrisLecar updated
        G.node[i] = {
            'model': 'MorrisLecar_a',
            'name': 'neuron_' + str(i),
            'extern': True, 
            'public': True, 
            'spiking': False, 
            'selector':'/lpu_1/out/gpot/' + str(i), 
            'V1': -1.2,
            'V2': 18.0,
            'V3': 2.0,
            'V4': 30.0,
            'V_l': -60.0, 
            'V_ca': 120.0, 
            'V_k': -84.0, 
            'G_l': 2.0, 
            'G_ca': 4.0, 
            'G_k': 8.0,
            'phi': 0.04, 
            'offset': 0.0,
            'initV': -50.0,
            'initn': 0.03
        }

    print 'writing'
    nx.write_gexf(G, 'simple_lpu_1.gexf.gz')
    print 'done'


input_num = 100

create_lpu_0(input_num)

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
I_max = -60
t = np.arange(0, dt*Nt, dt)
I = np.zeros((Nt, input_num), dtype=np.double)

#inputs current at points indicated
I[np.logical_and(t > start, t < stop)] = I_max

with h5py.File('simple_input.h5', 'w') as f: 
    f.create_dataset('array', (Nt, input_num), dtype = np.double, data = I)


