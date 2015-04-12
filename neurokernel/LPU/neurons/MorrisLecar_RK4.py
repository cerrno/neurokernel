__author__ = 'Lucas Schuermann'

from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os.path
import neurokernel.LPU.neurons as neurons

class MorrisLecar_RK4(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False):

        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.debug = debug
        self.cu_path = os.path.dirname(neurons.__file__)

        self.V = V

        self.n = garray.to_gpu(np.asarray(n_dict['initn'], dtype=np.float64))

        self.V_1 = garray.to_gpu(np.asarray(n_dict['V1'], dtype=np.float64))
        self.V_2 = garray.to_gpu(np.asarray(n_dict['V2'], dtype=np.float64))
        self.V_3 = garray.to_gpu(np.asarray(n_dict['V3'], dtype=np.float64))
        self.V_4 = garray.to_gpu(np.asarray(n_dict['V4'], dtype=np.float64))
        self.V_l = garray.to_gpu(np.asarray(n_dict['V_l'], dtype = np.float64))
        self.V_ca = garray.to_gpu(np.asarray(n_dict['V_ca'], dtype = np.float64))
        self.V_k = garray.to_gpu(np.asarray(n_dict['V_k'], dtype = np.float64))
        self.G_l = garray.to_gpu(np.asarray(n_dict['G_l'], dtype = np.float64))
        self.G_ca = garray.to_gpu(np.asarray(n_dict['G_ca'], dtype = np.float64))
        self.G_k = garray.to_gpu(np.asarray(n_dict['G_k'], dtype = np.float64))
        self.Tphi = garray.to_gpu(np.asarray(n_dict['phi'], dtype=np.float64))
        self.offset = garray.to_gpu(np.asarray(n_dict['offset'],
                                           dtype=np.float64))

        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'],
                     dtype=np.double))
        self.update = self.get_rk4_kernel()


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st, self.V, self.n.gpudata,
            self.num_neurons, self.I.gpudata, self.dt*1000,
            self.V_1.gpudata, self.V_2.gpudata, self.V_3.gpudata,
            self.V_4.gpudata, self.V_l.gpudata, self.V_ca.gpudata,
            self.V_k.gpudata, self.G_l.gpudata, self.G_ca.gpudata,
            self.G_k.gpudata, self.Tphi.gpudata, self.offset.gpudata)


    def get_rk4_kernel(self):
        template = open(self.cu_path+'/kernels/ML_RK4.cu')
        # Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
        # 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                           "nneu": self.update_block[0]},
                           options=["--ptxas-options=-v"])
        func = mod.get_function("morris_lecar_rk4")


        func.prepare([np.intp, np.intp, np.int32, np.intp, scalartype,
                      np.intp, np.intp, np.intp, np.intp,
                      np.intp, np.intp, np.intp, np.intp, np.intp,
                      np.intp, np.intp, np.intp])

        return func
