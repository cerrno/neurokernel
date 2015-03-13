#An implementation of a 2 neuron lpu that instantiates a Leaky and a Morrislecar 
#into a 1 edge/2node graph. Outputs to a .gefx file. Take from py notebooks. 

#matrix support
import numpy as np
#graph support
import networkx as nx

#input support
import h5py

#sets up neurons/networks
G = nx.DiGraph()

#FIGURE OUT HOW TO SET UP SECOND NODE HERE
G.add_nodes_from([0,1])
G.add_nodes_from([0,2])

G.node[0] = {
        'model' : 'LeakyIAF',
        'name' : 'neuron_0', 
        'extern': True, 
        'public': True, 
        'spiking': True, 
        'selector':'/a[0]', 
        'V': np.random.uniform(-0.06, -0.025),
        'Vr': -0.0675489770451,
        'Vt': -0.0251355161007,
        'R': 1.02445570216,
        'C':0.0669810502993
        }

G.node[1] = {
        'model' : 'MorrisLecar_RK4',
        #'model' : 'MorrisLecar', 
        'name' : 'neuron_1', 
        'extern': False, 
        'public': True, 
        'spiking': False, 
        'selector':'/a[1]', 
        'V1': -0.0012,
        'V2': 0.018,
        'V3': 0.002,
        'V4': 0.030,
        'V_l': -0.060, 
        'V_ca': 0.120, 
        'V_k': -0.084, 
        'G_l': 0.002, 
        'G_ca': 0.004, 
        'G_k': 0.008,
        'phi': 0.04, 
        'offset': 0,
        'initV': 0.0,
        'initn': 0.03
        }

#NOT GENERATING THIS NODE, FIGURE OUT WHY
G.node[2] = {
        'model' : 'MorrisLecar',
        'name' : 'neuron_2', 
        'extern': False, 
        'public': True, 
        'spiking': False, 
        'selector':'/a[2]', 
        'V1': 0.03,
        'V2': 0.015,
        'V3': 0,
        'V4': 0.03,
        'phi': 0.025, 
        'offset': 0,
        'initV': -0.05214,
        'initn': 0.03
        }

G.add_edge(0, 1, type='directed', attr_dict={
    'model': 'AlphaSynapse', 
    'name': 'synapse_0_1', 
    'class': 0,
    'ar': 1.1*1e2,
    'ad': 1.9*1e3,
    'reverse': 65*1e-3,
    'gmax': 2*1e-3,
    'conductance': True
    })

G.add_edge(0, 2, type='directed', attr_dict={
    'model': 'AlphaSynapse', 
    'name': 'synapse_0_2', 
    'class': 0,
    'ar': 1.1*1e2,
    'ad': 1.9*1e3,
    'reverse': 65*1e-3,
    'gmax': 2*1e-3,
    'conductance': True
    })

nx.write_gexf(G, 'simple_lpu.gexf.gz')


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
I_max = 0.6
t = np.arange(0, dt*Nt, dt)
I = np.zeros((Nt, 1), dtype=np.double)

#inputs current at points indicated
I[np.logical_and(t > start, t < stop)] = I_max

with h5py.File('simple_input.h5', 'w') as f: 
    f.create_dataset('array', (Nt, 1), dtype = np.double, data = I)


