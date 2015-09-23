# this is meant to be the cleanest prototype code for capturing images from
# a camera for use in the interface testing

# dependencies: video device connected via USB on port 0 (can modify below)
# libraries: opencv, (numpy), requisite dependencies and vanilla python dist

import cv2

camera_port = 0
throw_frames = 30

camera = cv2.VideoCapture(camera_port)

def get_image():
    retval, im = camera.read()
    return im

print("adjusting to light")
for i in xrange(throw_frames):
    temp = get_image()

print("saving image")
camera_capture = get_image()
file = "./test_image.png"
cv2.imwrite(file, camera_capture)

# we now have an image object returned from get_image that
# can be passed along to neurokernel
print(type(camera_capture))

del(camera)
