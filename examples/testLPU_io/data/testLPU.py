#An implementation of a 2 neuron lpu that instantiates a Leaky and a Morrislecar 
#into a 1 edge/2node graph. Outputs to a .gefx file. Take from py notebooks. 

#matrix support
import numpy as np
np.set_printoptions(threshold = 'nan')
#graph support
import networkx as nx

#input support
import h5py

<<<<<<< HEAD
def create_lpu_0():
=======
def create_lpu_0(neu_num):

>>>>>>> d87f8b04a1bff049f2dc4936d684b8c2c0eb3957
    #sets up neurons/networks
    G = nx.DiGraph()

    #sets up node connections in graph
<<<<<<< HEAD
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
            'model': 'MorrisLecar_a',
            'name': 'neuron_1',
            'extern': False, 
            'public': True, 
            'spiking': False, 
            'selector':'/lpu_1/out/gpot/0', 
=======
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
>>>>>>> d87f8b04a1bff049f2dc4936d684b8c2c0eb3957
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
<<<<<<< HEAD
            'initn': 0.03       
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
        'reverse'     : -0.08,
        'conductance' : True})



    nx.write_gexf(G, 'simple_lpu_1.gexf.gz')

=======
            'initn': 0.03
        }

    print 'writing'
    nx.write_gexf(G, 'simple_lpu_1.gexf.gz')
    print 'done'


input_num = 100

create_lpu_0(input_num)
>>>>>>> d87f8b04a1bff049f2dc4936d684b8c2c0eb3957

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
<<<<<<< HEAD
I_max = 60
t = np.arange(0, dt*Nt, dt)
I = np.zeros((Nt, 1), dtype=np.double)
=======
I_max = -60
t = np.arange(0, dt*Nt, dt)
I = np.zeros((Nt, input_num), dtype=np.double)
>>>>>>> d87f8b04a1bff049f2dc4936d684b8c2c0eb3957

#inputs current at points indicated
I[np.logical_and(t > start, t < stop)] = I_max

with h5py.File('simple_input.h5', 'w') as f: 
<<<<<<< HEAD
    f.create_dataset('array', (Nt, 1), dtype = np.double, data = I)

create_lpu_0()
=======
    f.create_dataset('array', (Nt, input_num), dtype = np.double, data = I)



>>>>>>> d87f8b04a1bff049f2dc4936d684b8c2c0eb3957
