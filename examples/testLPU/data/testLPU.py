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

#sets up node connections in graph
G.add_nodes_from([0,1])
G.add_nodes_from([0,2])
G.add_nodes_from([0,3])


#The default leaky that leads to all of the others
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

#MorrisLecar updated
G.node[1] = {
        'model' : 'MorrisLecar_RK4',
        'name' : 'neuron_1', 
        'extern': False, 
        'public': True, 
        'spiking': False, 
        'selector':'/a[1]', 
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

#MorrisLecar original
G.node[2] = {
        'model': 'MorrisLecar_a',
        'name': 'neuron_2',
        'extern': False, 
        'public': True, 
        'spiking': False, 
        'selector':'/a[2]', 
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

#HH Neuron Setup; LUCAS YOULL WANT TO LOOK AT THIS PART TO SET UP THE CONSTANTS
#I used the constants from matlab in neuroscience from my previous exercises

# Added updated constants from paper (Siciliano), but left initial volatge because I am too lazy to calcluate
# it again using the constants

G.node[3] = {
        'model': 'HodgkinHuxley_RK4',
        'name': 'neuron_3',
        'extern': False,
        'public': True,
        'spiking': True,
        'selector':'/a[3]',
        'initV': -50.0,
        'initn': 0.0003,
        'initm': 0.0011,
        'inith': 0.9998,
        'C_m': 0.01,
        'V_Na': 55.17,
        'V_K': -72.14,
        'V_l': -49.42,
        'g_Na': 1.0,
        'g_K': 0.36,
        'g_l': 0.003
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

G.add_edge(0, 3, type='directed', attr_dict={
    'model': 'AlphaSynapse',
    'name': 'synapse_0_3',
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


