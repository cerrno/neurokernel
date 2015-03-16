#!/usr/bin/env python

"""
Run timing test (non-GPU) scaled over number of LPUs.
"""

import numpy as np

import csv
import re
import subprocess
import sys

script_name = 'timing_demo.py'

w = csv.writer(sys.stdout)
for lpus in xrange(2, 9):
    average_step_sync_time_list = []
    average_throughput_list = []
    total_throughput_list = []
    runtime_list = []
    for i in xrange(3):
        out = subprocess.check_output(['srun', '-n', '1', '-c', str(lpus),
                                       'python', script_name,
                                       '-u', str(lpus), '-s', '1000', '-g', '0',
                                       '-m', '100'])
        average_step_sync_time, average_throughput, total_throughput, runtime = out.strip('()\n\"').split(', ')
        average_step_sync_time_list.append(float(average_step_sync_time))
        average_throughput_list.append(float(average_throughput))
        total_throughput_list.append(float(total_throughput))
        runtime_list.append(float(runtime))
    w.writerow([lpus,
                np.average(average_step_sync_time_list),
                np.average(average_throughput_list),
                np.average(total_throughput_list),
                np.average(runtime_list)])
