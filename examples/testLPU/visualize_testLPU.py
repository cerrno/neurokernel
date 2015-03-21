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
V.add_LPU('./data/simple_input.h5', LPU='Input')
V.add_plot({'type': 'waveform', 'ids': [[0]]}, 'input_Input')

#takes in the spike data/potential from the neuron output and plots it
V.add_LPU('simple_output_spike.h5',
          './data/simple_lpu.gexf.gz', 'Simple LPU (Spikes)')

#the [0,1] under ids should print both hh and leaky
V.add_plot({'type':'raster', 'ids': {0: [0]},
            'yticks': [0], 'yticklabels': [0]},
            'Simple LPU (Spikes)','Output')

V.add_plot({'type':'raster', 'ids': {0: [1]},
            'yticks': [0], 'yticklabels': [0]},
            'Simple LPU (Spikes)','Output')


V.add_LPU('simple_output_gpot.h5',
          './data/simple_lpu.gexf.gz', 'Simple LPU (Graded Potential)')

V.add_plot({'type': 'waveform', 'ids': {0:[0]}},
            'Simple LPU (Graded Potential)', 'Output')

V.add_plot({'type': 'waveform', 'ids': {0:[1]}},
            'Simple LPU (Graded Potential)', 'Output')



#vars for plots

#how often it updates
V._update_interval = 50

#rows and colums to plot in terms of size
V.rows = 4
V.cols = 1 

#self explantory
V.fontsize = 10
#V.out_filename = 'simple_output.avi'
#V.codec = 'libtheora'

#time step
V.dt = 0.0001

#Changes the sizes on the axis
V.xlim = [0, 1.0]
V.ylim = [-70.0, -50.0]

#figure size
V.figsize = (6, 4)

#runs the visualizer
V.run('simple_output.png', 120)

