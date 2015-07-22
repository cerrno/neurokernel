#@author: Amol Kapoor
#date: 3-13-15
#Visualizer for simpleLPU stuff

import matplotlib as mpl
mpl.use('agg')

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

# Temporary fix for bug in networkx 1.8:
nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}
#starts up the visualizer code
V = vis.visualizer()

#takes in the input file as defined in data, and plots it
V.add_LPU('simple_output_1_gpot.h5',
        './data/simple_lpu_1.gexf.gz', 'LPU2')

V.add_plot({'type': 'waveform', 'ids': {0:[1]}},
            'LPU2', 'Output')

V.add_plot({'type': 'waveform', 'ids': {0:[31]}},
            'LPU2', 'Output')

V.add_plot({'type': 'waveform', 'ids': {0:[51]}},
            'LPU2', 'Output')

V.add_plot({'type': 'waveform', 'ids': {0:[71]}},
            'LPU2', 'Output')

V.add_plot({'type': 'waveform', 'ids': {0:[91]}},
            'LPU2', 'Output')






#vars for plots

#how often it updates
V._update_interval = 50

#rows and colums to plot in terms of size
V.rows = 5
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

#runs the visualizer
V.run('simple_output_ports1.png', 120)

