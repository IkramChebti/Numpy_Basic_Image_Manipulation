import math
import cv2

import numpy as np

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `cv2.imread()` function - 
          whatch out  for the returned color format ! Check the following link for some fun : 
          https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    # Utilisez cv2.imread - le format RGB doit être retourné
    out=cv2.imread(image_path)
    ### VOTRE CODE ICI - FIN

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float32) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    imagesquare = np.square(image)
    out= np.multiply(imagesquare, 0.5)
    ### VOTRE CODE ICI - FIN

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: see if you can use  the opencv function `cv2.cvtColor()` 
    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT    
    out= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ### VOTRE CODE ICI - FIN

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    b,g,r= cv2.split(image)

    if channel == 'R' :
        out=r
    elif channel == 'G' :
        out=g
    elif channel == 'B' : 
        out=b
    ### VOTRE CODE ICI - FIN

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b= cv2.split(lab)
    if channel == 'L':
        out = l
    elif channel == 'A':
        out = a
    elif channel == 'B':
        out = b
    ### VOTRE CODE ICI - FIN

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    hsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v= cv2.split(hsv)
    if channel == 'H':
        out = h
    elif channel == 'S':
        out = s
    elif channel == 'V':
        out = v
    ### VOTRE CODE ICI - FIN

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### VOTRE CODE ICI - DEBUT
    h1,w1, channels1= image1.shape
    half1= w1//2
    h2,w2, channels2= image2.shape
    half2= w2//2
    imageleft = image1[:,:half1]
    imageright = image2[:,half2:]
    imageleft = rgb_exclusion(imageleft,channel1)
    imageright = rgb_exclusion(imageright,channel2)
    out = np.concatenate((imageleft, imageright), axis=1)
    ### VOTRE CODE ICI - FIN

    return out


def mix_quadrants(image):
    """
    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    h,w= image.shape[:2]
    cX= w//2
    cY= h//2
    Top_left_quadrant = image[0:cY, 0:cX]
    Top_right_quadrant = image[0:cY, cX:w]
    Bottom_left_quadrant = image[cY:h, 0:cX]
    Bottom_right_quadrant = image[cY:h, cX:w]
    Quadrant_supérieur_gauche = rgb_exclusion(Top_left_quadrant,'R')
    Quadrant_supérieur_droit = dim_image(Top_right_quadrant)
    Quadrant_inférieur_gauche = np.power(Bottom_left_quadrant, 0.5)
    Quadrant_inférieur_droit = rgb_exclusion(Bottom_right_quadrant,'R')
    half1 = np.concatenate((Quadrant_supérieur_gauche,Quadrant_inférieur_gauche), axis =1)
    half2 = np.concatenate((Quadrant_supérieur_droit,Quadrant_inférieur_droit), axis =1)
    out = np.concatenate((half1, half2), axis=0)
    ### VOTRE CODE ICI - FIN

    return out
