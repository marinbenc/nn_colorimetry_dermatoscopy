# This code is originally written by Mina Sami and Mahmoud Khaled
# https://github.com/MinaSGorgi/Color-Constancy/blob/028d0f8e0c56bd3b32e958861a6e000ab07b56fe/voters.py#L17

import numpy as np
import matplotlib.pyplot as plt 

import skimage
from skimage import filters
import cv2 as cv

def shades_of_grey_wb(image, p):
    """
    Applies Shades-of-Grey color constancy algorithm to correct the input image.
    The input image is expected to be in RGB.
    """
    sog_illum = grey_edge(image, njet=0, mink_norm=p, sigma=0)
    img = correct_image(image, sog_illum)
    return img

def grey_edge(image, njet=0, mink_norm=1, sigma=1):
    """
    Estimates the light source of an input_image as proposed in:
    J. van de Weijer, Th. Gevers, A. Gijsenij
    "Edge-Based Color Constancy"
    IEEE Trans. Image Processing, accepted 2007.
    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,
    Shades-of-Grey or Grey-Edge algorithm.

    :param image: rgb input image (NxMx3)
    :param njet: the order of differentiation (range from 0-2)
    :param mink_norm: minkowski norm used (if mink_norm==-1 then the max
           operation is applied which is equal to minkowski_norm=infinity).
    :param sigma: sigma used for gaussian pre-processing of input image

    :return: illuminant color estimation

    :raise: ValueError
    """

    # pre-process image by applying gauss filter
    gauss_image = filters.gaussian(image, sigma=sigma, channel_axis=2)

    # get njet-order derivative of the pre-processed image
    if njet == 0:
        deriv_image = [gauss_image[:, :, channel] for channel in range(3)]
    else:   
        if njet == 1:
            deriv_filter = filters.sobel
        elif njet == 2:
            deriv_filter = filters.laplace
        else:
            raise ValueError("njet should be in range[0-2]! Given value is: " + str(njet))     
        deriv_image = [np.abs(deriv_filter(gauss_image[:, :, channel])) for channel in range(3)]

    # remove saturated pixels in input image
    for channel in range(3):
        deriv_image[channel][image[:, :, channel] >= 255] = 0.

    # estimate illuminations
    if mink_norm == -1:  # mink_norm = inf
        estimating_func = np.max 
    else:
        estimating_func = lambda x: np.power(np.sum(np.power(x, mink_norm)), 1 / mink_norm)
    illum = [estimating_func(channel) for channel in deriv_image]

    # normalize estimated illumination
    som = np.sqrt(np.sum(np.power(illum, 2)))
    illum = np.divide(illum, som)

    return illum


def correct_image(image, illum):
    """
    Corrects image colors by performing diagonal transformation according to 
    given estimated illumination of the image.
    
    :param image: rgb input image (NxMx3)
    :param illum: estimated illumination of the image

    :return: corrected image
    """

    correcting_illum = illum * np.sqrt(3)
    corrected_image = image / 255.

    for channel in range(3):
        corrected_image[:, :, channel] /= correcting_illum[channel]
    return np.clip(corrected_image, 0., 1.)

if __name__ == "__main__":
    # test
    img = cv.imread("mskcc_processed/ISIC_0077599.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    corrected_img = shades_of_grey_wb(img, p=6)

    plt_img = np.hstack((img/255., corrected_img))

    plt.imshow(plt_img)
    plt.savefig("temp.png")
    #plt.show()