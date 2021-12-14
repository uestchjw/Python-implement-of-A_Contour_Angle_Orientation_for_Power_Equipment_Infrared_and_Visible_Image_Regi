
from hjw_graymosaic import hjw_graymosaic
import numpy as np
def hjw_rgbmosaic(I1,I2,affmat):
    # % CP_MOSAIC
    # % input: a pair of source rgb images
    # % output: a mosaic image based on liniear interplation transformation

    return np.dstack(  (hjw_graymosaic(I1[:,:,0],I2[:,:,0],affmat)\
                        ,hjw_graymosaic(I1[:,:,1],I2[:,:,1],affmat)\
                        ,hjw_graymosaic(I1[:,:,2],I2[:,:,2],affmat)  
                        )  )
