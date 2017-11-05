#######################################
#rescale and invert color (if desired)#
########################################################################

import numpy as np

def rescale_data(data, low=0.1, hi=1):
    #rescaling and inverting images
    #https://www.mathworks.com/help/vision/ref/contrastadjustment.html
    #Since maxpooling is used, we want the interesting stuff (craters) to be 1, not 0.
    #But ignore null background pixels, keep them at 0.
    for img in data:
        minn, maxx = np.min(img[img>0]), np.max(img[img>0])
        img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    return data

