__author__ = 'Lucas Schuermann'

from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class HodgkinHuxley_RK4(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False):

        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.debug = debug

        self.V = V

        self.n = garray.to_gpu(np.asarray(n_dict['initn'], dtype=np.float64))

        # LVS TODO: HH needed arguments
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
        # LVS TODO: match call to final parameters needed by HH implementation
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st, self.V, self.n.gpudata,
            self.num_neurons, self.I.gpudata, self.dt)#, self.steps,
            #self.V_1.gpudata, self.V_2.gpudata, self.V_3.gpudata,
            #self.V_4.gpudata, self.Tphi.gpudata, self.offset.gpudata)


    def get_HH_rk4_kernel(self):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL


    // constant defines
    #define CONST 0

    // device compute submethods
    __device__ %(type)s compute_dV(%(type)s V, %(type)s n, %(type)s m, %(type)s h,
                                    %(type) I, %(type)s C_m, %(type)s V_Na,
                                    %(type)s V_K,  %(type)s V_l,  %(type)s g_Na,
                                    %(type)s g_K,  %(type)s g_l)
    {
        return (-1/C_m) * (g_K*n*n*n*n(V-V_K) + g_Na*m*m*m*h*(V-V_Na)
                + g_l*(V-V_l)-I);
    }

    __device__ %(type)s compute_dn(%(type)s V, %(type)s n)
    {
        %(type)s alpha_n = (0.01*(V+10.0)) / (exp((V+10.0)/10.0) - 1);
        %(type)s beta_n = 0.125 * exp(V/80.0);
        return alpha_n * (1-n) - beta_n * n;
    }

    __device__ %(type)s compute_dm(%(type)s V, %(type)s m)
    {
        %(type)s alpha_m = (0.1*(V+25.0)) / (exp(V+25.0)/10.0) -1);
        %(type)s beta_m = 4 * exp(V/18.0);
        return alpha_m * (1-m) - beta_m * m;
    }

    __device__ %(type)s compute_dh(%(type)s V, %(type)s h)
    {
        %(type)s alpha_h = 0.07 * exp(V/20.0);
        %(type)s beta_h = 1 / (exp((V+30.0)/10.0) + 1);
        return alpha_h * (1-h) - beta_h * h;
    }

    // main kernel
    __global__ void
    hodgkin_huxley_rk4(%(type)s* g_V, %(type)s* I_pre, %(type)s dt)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        %(type)s I, V;

        if(cart_id < num_neurons)
        {
            V = g_V[cart_id];
            I = I_pre[cart_id];

            %(type)s dV, dn, dm, dh;

            %(type)s alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;

            // auxiliary quantities
            //alpha_n = (0.01*(V+10.0)) / (exp((V+10.0)/10.0) - 1);
            //beta_n = 0.125 * exp(V/80.0);
            //alpha_m = (0.1*(V+25.0)) / (exp(V+25.0)/10.0) -1);
            //beta_m = 4 * exp(V/18.0);
            //alpha_h = 0.07 * exp(V/20.0);
            //beta_h = 1 / (exp((V+30.0)/10.0) + 1);

            %(type)s k1_V, k2_V, k3_V, k4_V;
            %(type)s k1_n, k2_n, k3_n, k4_n;
            %(type)s k1_m, k2_m, k3_m, k4_m;
            %(type)s k1_h, k2_h, k3_h, k4_h;

            // RK4 using derivative calculations
            // LVS TODO: Could storing the values of x+0.5*k1_x
            // help to optimize the program?
            k1_V = dt * compute_dV(V, n, m, h,
                                    I, C_m[cart_id], V_Na[cart_id],
                                    V_K[cart_id], V_l[cart_id], g_Na[cart_id],
                                    g_K[cart_id], g_l[cart_id]);
            k1_n = dt * compute_dn(V, n);
            k1_m = dt * compute_dm(V, m);
            k1_h = dt * compute_dh(V, h);

            k2_V = dt * compute_dV(V+0.5*k1_V, n+0.5*k1_n, m+0.5*k1_m, h+0.5*k1_h,
                                    I, C_m[cart_id], V_Na[cart_id],
                                    V_K[cart_id], V_l[cart_id], g_Na[cart_id],
                                    g_K[cart_id], g_l[cart_id]);
            k2_n = dt * compute_dn(V+0.5*k1_V, n+0.5*k1_n);
            k2_m = dt * compute_dm(V+0.5*k1_V, m+0.5*k1_m);
            k2_h = dt * compute_dh(V+0.5*k1_V, h+0.5*k1_h);

            k3_V = dt * compute_dV(V+0.5*k2_V, n+0.5*k2_n, m+0.5*k2_m, h+0.5*k2_h,
                                    I, C_m[cart_id], V_Na[cart_id],
                                    V_K[cart_id], V_l[cart_id], g_Na[cart_id],
                                    g_K[cart_id], g_l[cart_id]);
            k3_n = dt * compute_dn(V+0.5*k2_V, n+0.5*k2_n);
            k3_m = dt * compute_dm(V+0.5*k2_V, m+0.5*k2_m);
            k3_h = dt * compute_dh(V+0.5*k2_V, h+0.5*k2_h);

            k4_V = dt * compute_dV(V+k3_V, n+k3_n, m+k3_m, h+k3_h,
                                    I, C_m[cart_id], V_Na[cart_id],
                                    V_K[cart_id], V_l[cart_id], g_Na[cart_id],
                                    g_K[cart_id], g_l[cart_id]);
            k4_n = dt * compute_dn(V+k3_V, n+k3_n);
            k4_m = dt * compute_dm(V+k3_V, m+k3_m);
            k4_h = dt * compute_dh(V+k3_V, h+k3_h);

            // "return" quantities
            V += (k1_V + 2*(k2_V + k3_V) + k4_V)/6.0;
            n += (k1_n + 2*(k2_n + k3_n) + k4_n)/6.0;
            m += (k1_m + 2*(k2_m + k3_m) + k4_m)/6.0;
            h += (k1_h + 2*(k2_h + k3_h) + k4_h)/6.0;
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
        func = mod.get_function("hodgkin_huxley_rk4")

        # LVS TODO: ensure match for above newly specified HH arguments
        func.prepare([np.intp, np.intp, np.int32, np.intp, scalartype,
                      np.int32, np.intp, np.intp, np.intp,
                      np.intp, np.intp, np.intp])


        return func