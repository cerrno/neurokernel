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