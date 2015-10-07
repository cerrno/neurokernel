__author__ = 'Lucas Schuermann'

from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os.path
import neurokernel.LPU.neurons as neurons


class HodgkinHuxley_RK4(BaseNeuron):
    '''
    LVS Hodgkin-Huxley neural model
    Runge-Kutta 4th order multivariable integration scheme

    Note: Requires shift of constants with initial voltage of -65 mV
    Recommended time step of 2e-5 or less

    Quantities
        initV:  membrane potential voltage
        initn
        initm
        inith

    Constants:
        C_m:    membrane capacitance per unit area
        V_Na:   sodium ion channel potential
        V_K:    potassium ion channel potential
        V_l:    leak channel potential
        g_Na:   electrical conductance of voltage-gated sodium ion channel
        g_K:    electrical conductance of voltage-gated potassium ion channel
        g_l:    electrical conductance of leak channel
    '''
    def __init__(self, n_dict, V, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.debug = debug
        self.LPU_id = LPU_id
        self.cu_path = os.path.dirname(neurons.__file__)

        self.V = V

        self.m = garray.to_gpu(np.asarray(n_dict['initm'], dtype=np.float64))
        self.h = garray.to_gpu(np.asarray(n_dict['inith'], dtype=np.float64))

        self.C_m = garray.to_gpu(np.asarray(n_dict['C_m'], dtype=np.float64))
        self.V_Na = garray.to_gpu(np.asarray(n_dict['V_Na'], dtype=np.float64))
        self.V_K = garray.to_gpu(np.asarray(n_dict['V_K'], dtype=np.float64))
        self.V_l = garray.to_gpu(np.asarray(n_dict['V_l'], dtype=np.float64))
        self.g_Na = garray.to_gpu(np.asarray(n_dict['g_Na'], dtype=np.float64))
        self.g_K = garray.to_gpu(np.asarray(n_dict['g_K'], dtype=np.float64))
        self.g_l = garray.to_gpu(np.asarray(n_dict['g_l'], dtype=np.float64))

        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'], dtype=np.double))
        self.update = self.get_HH_rk4_kernel()


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st,
            self.V, self.n.gpudata, self.m.gpudata, self.h.gpudata,
            self.num_neurons, self.I.gpudata, self.dt*1000,
            self.C_m.gpudata, self.V_Na.gpudata, self.V_K.gpudata,
            self.V_l.gpudata, self.g_Na.gpudata, self.g_K.gpudata,
            self.g_l.gpudata)


    def get_HH_rk4_kernel(self):
        template = open(self.cu_path+'/kernels/HH_RK4.cu').read()
        # Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0],
        # 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                           "nneu": self.update_block[0]},
                           options=["--ptxas-options=-v"])
        func = mod.get_function("hodgkin_huxley_rk4")

        func.prepare('PPPPiP'+np.dtype(dtype).char+'PPPPPPP')
        #             [np.intp, np.intp, np.intp, np.intp,
        #              np.int32, np.intp, scalartype,
        #              np.intp, np.intp, np.intp,
        #              np.intp, np.intp, np.intp,
        #              np.intp])
        return func
