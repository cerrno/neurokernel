'''
Simple script to read each frame of a h5py image buffer and visualize output
'''

import cv2
import neurokernel.LPU.utils.simpleio as si

import numpy as np


filename = 'imagedata'

a = si.read_array(filename)

print a

for i in range(0, 1000) :

    b = a[i]

    b = b.astype(np.uint8)

    print b 

    #    cv2.imshow('image', b)
    #    cv2.waitKey(30)

#cv2.destroyAllWindows()

