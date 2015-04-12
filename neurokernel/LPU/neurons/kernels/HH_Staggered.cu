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
hodgkin_huxley_rk4(%(type)s* g_V, %(type)s* g_n, %(type)s* g_m, %(type)s* g_h,
                    int num_neurons, %(type)s* I_pre, %(type)s dt,
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

        %(type)s k1_V, k2_V, k3_V, k4_V;
        %(type)s k1_n, k2_n, k3_n, k4_n;
        %(type)s k1_m, k2_m, k3_m, k4_m;
        %(type)s k1_h, k2_h, k3_h, k4_h;

        // RK4 using device derivative calculation functions

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

        // compute new quantities
        V += (k1_V + 2*(k2_V + k3_V) + k4_V)/6.0;
        n += (k1_n + 2*(k2_n + k3_n) + k4_n)/6.0;
        m += (k1_m + 2*(k2_m + k3_m) + k4_m)/6.0;
        h += (k1_h + 2*(k2_h + k3_h) + k4_h)/6.0;

        g_V[cart_id] = V;
        g_n[cart_id] = n;
        g_m[cart_id] = m;
        g_h[cart_id] = h;
    }
}