

import math
import numpy as np



# 这个函数求的是直线的角度
def hjw_atan(dy,dx):
    # % CP_ATAN calculate the Inclination angle of one line
    # 作者是n行1列的list的tanvalue
    # 但是我是n行1列的numpy
    tanvalue = np.zeros((dy.shape[0],1))
    for i in range(dy.shape[0]):
        if dy[i] == 0 and dx[i] == 0:
            tanvalue[i] = 90
            continue
        else:
            tanvalue[i] = round(180/math.pi * math.atan(dy[i]/dx[i]))


    return tanvalue
