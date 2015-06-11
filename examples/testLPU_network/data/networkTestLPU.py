#!/usr/bin/env python

"""
Create generic LPU and simple pulse input signal.
"""

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx

def create_lpu(file_name, lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model. 

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.
    """
    
    # Set numbers of neurons:
    neu_type = ('sensory', 'local', 'proj')
    neu_num = (N_sensory, N_local, N_proj)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.DiGraph()
    G.add_nodes_from(range(sum(neu_num)))

    idx = 0
    spk_out_id = 0
    gpot_out_id = 0
    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            name = t+"_"+str(i)
            G.node[idx] = {
                'model': 'HodgkinHuxley_RK4',
                'name': name+'_g',
                'extern': True if t == 'sensory' else False,
                'public': True if t == 'proj' else False,
                'spiking': False,
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

            # Projection neurons are all assumed to be attached to output
            # ports (which are not represented as separate nodes):
            if t == 'proj':
                G.node[idx]['selector'] = '/%s/out/gpot/%s' % (lpu_name, str(gpot_out_id))
                gpot_out_id += 1
            
            idx += 1

    # Assume a probability of synapse existence for each group of synapses:
    # sensory -> local, sensory -> projection, local -> projection, 
    # projection -> local:            
    for r, (i, j) in zip((0.5, 0.1, 0.1, 0.3),
                         ((0, 1), (0, 2), (1, 2), (2, 1))):
        src_off = sum(neu_num[0:i])
        tar_off = sum(neu_num[0:j])
        for src, tar in product(range(src_off, src_off+neu_num[i]),
                                range(tar_off, tar_off+neu_num[j])):

            # Don't connect all neurons:
            if np.random.rand() > r: continue

            # Connections from the sensory neurons use the alpha function model;
            # all other connections use the power_gpot_gpot model:
            name = G.node[src]['name'] + '-' + G.node[tar]['name']
           
            G.add_edge(src,tar,type='directed',attr_dict={
                'model'       : 'power_gpot_gpot_sig',
                'name'        : name,
                'class'       : 2 if G.node[tar]['spiking'] is True else 3,
                'slope'       : 0.8,
                'threshold'   : -45,
                'power'       : 10.0,
                'saturation'  : 30.0,
                'delay'       : 1.0,
                'reverse'     : -0.08,
                'conductance' : True})

    nx.write_gexf(G, file_name)

def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t>start, t<stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='generic_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='generic_input.h5',
                        help='Input file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    parser.add_argument('-l', '--lpu', type=str, default='gen',
                        help='LPU name')

    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 1.0
    start = 0.3
    stop = 0.6
    I_max = 0.6
    neu_num = [np.random.randint(31, 40) for i in xrange(3)]
    print neu_num
    create_input(args.in_file_name, neu_num[0], dt, dur, start, stop, I_max)
    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
