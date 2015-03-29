__author__ = 'Lucas Schuermann'

from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class HodgkinHuxley_Euler(BaseNeuron):
    '''
    LVS Hodgkin-Huxley neural model
    Interpolated Euler integration scheme

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
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.LPU_id = LPU_id
        self.debug = debug

        self.ddt = dt / self.steps

        self.V = V

        self.n = garray.to_gpu(np.asarray(n_dict['initn'], dtype=np.float64))
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
        self.update = self.get_HH_euler_kernel()


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st,
            self.V, self.n.gpudata, self.m.gpudata, self.h.gpudata,
            self.num_neurons, self.I.gpudata, self.ddt*1000, self.steps,
            self.C_m.gpudata, self.V_Na.gpudata, self.V_K.gpudata,
            self.V_l.gpudata, self.g_Na.gpudata, self.g_K.gpudata,
            self.g_l.gpudata)


    def get_HH_euler_kernel(self):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL

    // device compute submethods
    __device__ %(type)s compute_dV(%(type)s V, %(type)s n, %(type)s m, %(type)s h,
                                    %(type)s I, %(type)s C_m, %(type)s V_Na,
                                    %(type)s V_K,  %(type)s V_l,  %(type)s g_Na,
                                    %(type)s g_K,  %(type)s g_l)
    {
        return (-1/C_m) * (g_K*n*n*n*n*(V-V_K) + g_Na*m*m*m*h*(V-V_Na) + g_l*(V-V_l)-I);
    }

    __device__ %(type)s compute_dn(%(type)s V, %(type)s n)
    {
        // LVS TODO: This seems hacky
        %(type)s alpha_n;
        if(V == -55.0)
            alpha_n = 0.1;
        else
            alpha_n = (-0.01*(55.0+V)) / (exp(-(55.0+V)/10.0)-1);
        %(type)s beta_n = 0.125 * exp(-(V+65)/80.0);
        return (alpha_n * (1-n) - beta_n * n);
    }

    __device__ %(type)s compute_dm(%(type)s V, %(type)s m)
    {
        %(type)s alpha_m;
        if(V == -40.0)
            alpha_m = 1.0;
        else
            alpha_m = (-0.1*(40+V)) / (exp(-(40+V)/10.0) - 1);
        %(type)s beta_m = 4 * exp(-(V+65)/18.0);
        return (alpha_m * (1-m) - beta_m * m);
    }

    __device__ %(type)s compute_dh(%(type)s V, %(type)s h)
    {
        %(type)s alpha_h = 0.07 * exp(-(V+65)/20.0);
        %(type)s beta_h = 1 / (exp(-(35+V)/10.0) + 1);
        return (alpha_h * (1-h) - beta_h * h);
    }

    // main kernel
    __global__ void
    hodgkin_huxley_euler(%(type)s* g_V, %(type)s* g_n, %(type)s* g_m, %(type)s* g_h,
                        int num_neurons, %(type)s* I_pre, %(type)s dt, int nsteps,
                        %(type)s* C_m, %(type)s* V_Na, %(type)s* V_K,
                        %(type)s* V_l, %(type)s* g_Na, %(type)s* g_K,
                        %(type)s* g_l)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        %(type)s I, V, n, m, h;

        if(cart_id < num_neurons)
        {
            I = I_pre[cart_id];
            V = g_V[cart_id];
            n = g_n[cart_id];
            m = g_m[cart_id];
            h = g_h[cart_id];

            %(type)s dV, dn, dm, dh;

            for(int i = 0; i < nsteps; ++i)
            {
                dV = compute_dV(V, n, m, h,
                                    I, C_m[cart_id], V_Na[cart_id],
                                    V_K[cart_id], V_l[cart_id], g_Na[cart_id],
                                    g_K[cart_id], g_l[cart_id]);
                dn = compute_dn(V, n);
                dm = compute_dm(V, m);
                dh = compute_dh(V, h);

                V += dt * dV;
                n += dt * dn;
                m += dt * dm;
                h += dt * dh;
            }

            g_V[cart_id] = V;
            g_n[cart_id] = n;
            g_m[cart_id] = m;
            g_h[cart_id] = h;
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
        func = mod.get_function("hodgkin_huxley_euler")

        func.prepare([np.intp, np.intp, np.intp, np.intp,
                      np.int32, np.intp, scalartype, np.int32,
                      np.intp, np.intp, np.intp,
                      np.intp, np.intp, np.intp,
                      np.intp])


        return func
