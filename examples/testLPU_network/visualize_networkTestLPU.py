#!/usr/bin/env python

"""
Visualize generic LPU demo output.

Notes
-----
Generate demo output by running

python generic_demo.py
"""

import numpy as np
import matplotlib as mpl
mpl.use('agg')

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

# Select IDs of projection neurons:
G = nx.read_gexf('./data/generic_lpu.gexf.gz')
neu_proj = sorted([int(k) for k, n in G.node.items() if n['name'][:4] == 'proj'])

print neu_proj

V = vis.visualizer()
V.add_LPU('./data/generic_input.h5', LPU='Sensory')
V.add_plot({'type':'waveform', 'ids': [[0]]}, 'input_Sensory')

V.add_LPU('generic_output_gpot.h5',
          './data/generic_lpu.gexf.gz', 'Generic LPU')

V.add_plot({'type':'waveform', 'ids': {0:[0]}},'Generic LPU','Output')
V.add_plot({'type':'waveform', 'ids': {0:[20]}},'Generic LPU','Output')
V.add_plot({'type':'waveform', 'ids': {0:[40]}},'Generic LPU','Output')
V.add_plot({'type':'waveform', 'ids': {0:[60]}},'Generic LPU','Output')
V.add_plot({'type':'waveform', 'ids': {0:[80]}},'Generic LPU','Output')
V.add_plot({'type':'waveform', 'ids': {0:[90]}},'Generic LPU','Output')


#how often it updates
V._update_interval = 50

#rows and colums to plot in terms of size
V.rows = 7
V.cols = 1 

#self explantory
V.fontsize = 10
#V.out_filename = 'simple_output.avi'
#V.codec = 'libtheora'

#time step
V.dt = 0.0001

#Changes the sizes on the axis
V.xlim = [0, 1.0]
V.ylim = [-70.0, 10.0]

#figure size
V.figsize = (16, 9)

V.title = "Simple LPU Testing RK4 Models"

#runs the visualizer
V.run('simple_output_network.png', 120)




