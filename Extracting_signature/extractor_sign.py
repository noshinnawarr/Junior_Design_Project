import cv2
from skimage import measure, morphology
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np

def extract_signature(source_image):
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100
    constant_parameter_4 = 18

    img = cv2.threshold(source_image, 127, 255, cv2.THRESH_BINARY)[1]
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    total_area = 0
    counter = 0
    the_biggest_component = 0

    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250:
            if region.area > the_biggest_component:
                the_biggest_component = region.area

    average = total_area / counter
    a4_small = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big = a4_small * constant_parameter_4

    pre_version = morphology.remove_small_objects(blobs_labels, a4_small)
    component_sizes = np.bincount(pre_version.ravel())
    too_big = component_sizes > a4_big
    too_big_mask = too_big[pre_version]
    pre_version[too_big_mask] = 0

    plt.imsave('pre_version.png', pre_version)
    img = cv2.imread('pre_version.png', 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return img
