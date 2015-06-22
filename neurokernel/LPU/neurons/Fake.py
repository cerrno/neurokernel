from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class Fake(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False):

        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug

        self.ddt = dt / self.steps

        self.V = V

        self.n = garray.to_gpu(np.asarray(n_dict['initn'], dtype=np.float64))

        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'], 
                         dtype=np.double))
        self.update = self.get_fake()


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st, self.V, self.n.gpudata, 
            self.num_neurons, self.I.gpudata, self.ddt*1000, self.steps)


    def get_fake(self):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL

    __global__ void
    fake(%(type)s* g_V, %(type)s* g_n, int num_neurons, 
                       %(type)s* I_pre, %(type)s dt, int nsteps)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        %(type)s I, V, n;

        if(cart_id < num_neurons)
        {
            V = g_V[cart_id];
            I = I_pre[cart_id];
            n = g_n[cart_id];

            g_V[cart_id] = V;
            g_n[cart_id] = n;
        }

    }
    """ 
    # Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
    # 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                           "nneu": self.update_block[0]}, 
                           options=["--ptxas-options=-v"])
        func = mod.get_function("fake")


        func.prepare([np.intp, np.intp, np.int32, np.intp, scalartype, 
                      np.int32])
        return func
