#define NVAR 2
#define NNEU %(nneu)d //NROW * NCOL


__device__ %(type)s compute_n(%(type)s V, %(type)s n, %(type)s V_3, %(type)s V_4, %(type)s Tphi)
{
    %(type)s n_inf = 0.5 * (1 + tanh((V - V_3) / V_4));
    %(type)s dn = Tphi * cosh(( V - V_3) / (V_4*2)) * (n_inf - n);
    return dn;
}

__device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, %(type)s V_1, %(type)s V_2, %(type)s V_L, %(type)s V_Ca, %(type)s V_K, %(type)s g_L, %(type)s g_Ca, %(type)s g_K, %(type)s offset)
{
    %(type)s m_inf = 0.5 * (1+tanh((V - V_1)/V_2));
    %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset);
    return dV;
}


__global__ void
morris_lecar_rk4(%(type)s* g_V, %(type)s* g_n, int num_neurons,
                   %(type)s* I_pre, %(type)s dt,
                   %(type)s* V_1, %(type)s* V_2, %(type)s* V_3,
                   %(type)s* V_4, %(type)s* V_L, %(type)s* V_Ca,
                   %(type)s* V_K, %(type)s* g_L, %(type)s* g_Ca,
                   %(type)s* g_K, %(type)s* Tphi, %(type)s* offset)
{
    int bid = blockIdx.x;
    int cart_id = bid * NNEU + threadIdx.x;

    %(type)s I, V, n;

    if(cart_id < num_neurons)
    {
        V = g_V[cart_id];
        I = I_pre[cart_id];
        n = g_n[cart_id];

        %(type)s k1_V, k1_n, k2_V, k2_n, k3_V, k3_n, k4_V, k4_n;

        // LVS TODO: look into storing dereferenced array values
        k1_V = dt * compute_V(V, n, I,
                            V_1[cart_id], V_2[cart_id], V_L[cart_id],
                            V_Ca[cart_id], V_K[cart_id], g_L[cart_id],
                            g_Ca[cart_id], g_K[cart_id], offset[cart_id]);
        k1_n = dt * compute_n(V, n, V_3[cart_id],
                            V_4[cart_id], Tphi[cart_id]);
        k2_V = dt * compute_V(V + 0.5*k1_V, n + 0.5*k1_n, I,
                            V_1[cart_id], V_2[cart_id], V_L[cart_id],
                            V_Ca[cart_id], V_K[cart_id], g_L[cart_id],
                            g_Ca[cart_id], g_K[cart_id], offset[cart_id]);
        k2_n = dt * compute_n(V + 0.5*k1_V, n + 0.5*k1_n, V_3[cart_id],
                            V_4[cart_id], Tphi[cart_id]);
        k3_V = dt * compute_V(V + 0.5*k2_V, n + 0.5*k2_n, I,
                            V_1[cart_id], V_2[cart_id], V_L[cart_id],
                            V_Ca[cart_id], V_K[cart_id], g_L[cart_id],
                            g_Ca[cart_id], g_K[cart_id], offset[cart_id]);
        k3_n = dt * compute_n(V + 0.5*k2_V, n + 0.5*k2_n, V_3[cart_id],
                            V_4[cart_id], Tphi[cart_id]);
        k4_V = dt * compute_V(V + k3_V, n + k3_n, I,
                            V_1[cart_id], V_2[cart_id], V_L[cart_id],
                            V_Ca[cart_id], V_K[cart_id], g_L[cart_id],
                            g_Ca[cart_id], g_K[cart_id], offset[cart_id]);
        k4_n = dt * compute_n(V + k3_V, n + k3_n, V_3[cart_id],
                            V_4[cart_id], Tphi[cart_id]);
        V += (k1_V + 2*(k2_V + k3_V) + k4_V)/6.0;
        n += (k1_n + 2*(k2_n + k3_n) + k4_n)/6.0;

        g_V[cart_id] = V;
        g_n[cart_id] = n;
    }

}