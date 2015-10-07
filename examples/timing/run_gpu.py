#!/usr/bin/env python

"""
Run timing test (GPU) scaled over number of ports.
"""

import csv
import glob
import multiprocessing as mp
import os
import re
import subprocess
import sys

import numpy as np

from neurokernel.tools.misc import get_pids_open

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

out_file = sys.argv[1]
script_name = 'timing_demo_gpu.py'
trials = 3
lpus = 2

def check_and_print_output(*args):
    for i in xrange(5):
        # CUDA < 7.0 doesn't properly clean up IPC-related files; since
        # these can cause problems, we manually remove them before launching
        # each job:
        ipc_files = glob.glob('/dev/shm/cuda.shm*')
        for ipc_file in ipc_files:

            # Only remove files that are not being held open by any processes:
            if not get_pids_open(ipc_file):
                try:
                    os.remove(ipc_file)
                except:
                    pass

        try:
            out = subprocess.check_output(*args, env=os.environ, stderr=DEVNULL)
        except Exception as e:
            out = e.output
        if 'error' not in out:
            break
    print out,
    return out

pool = mp.Pool(1)
results = []
for spikes in np.linspace(50, 15000, 25, dtype=int):
    for i in xrange(trials):
        r = pool.apply_async(check_and_print_output, 
                             [['srun', '-n', '1', '-c', str(lpus+2),
                               '-p', 'huxley',
                               '--gres=gpu:%s' % lpus,
                               'python', script_name,
                               '-u', str(lpus), '-s', str(spikes),
                               '-g', '0', '-m', '50']])
        results.append(r)
f = open(out_file, 'w', 0)
w = csv.writer(f)
for r in results:
    w.writerow(r.get().strip('[]\n\"').split(', '))
f.close()
