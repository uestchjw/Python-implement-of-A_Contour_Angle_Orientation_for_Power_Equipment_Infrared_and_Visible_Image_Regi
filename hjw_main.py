# infrared and visible images registration,  The whole work is divided into 5 parts

# 带着问题：
    # 1.descriptor和主方向angle有什么区别
    # 2.如果说最后只要一个 transform matrix 即可，那么为什么要那么多对匹配点呢?
    # 3.为什么一定要找到curve再算主方向呢？为什么不是curve就不行呢？   短的canny检测出来的离散点，她认为不算做curve

# 基本库
import numpy as np
import matplotlib.pyplot as plt

# 自定义函数
from hjw_readimage import *
from hjw_resizeImage import *
from hjw_registration import hjw_registration
from hjw_getAffine import hjw_getAffine

from hjw_subpixelFine import hjw_subpixelFine
from hjw_graymosaic import hjw_graymosaic
from hjw_rgbmosaic import hjw_rgbmosaic

if __name__ == '__main__':

    # source image path
    path_infrared = 'D:/Neural Network/Sun_liguo/Example_Images/infrared_images/'
    path_visible = 'D:/Neural Network/Sun_liguo/Example_Images/visible_images/'

    img_infrared = 'I3.jpg'
    img_visible = 'V3.jpg'

    # %%
    # section 1: Read source images 
    # 这里省略了张正友畸变校正法，去除红外图片的畸变
    I1_rgb,I1_gray,I2_rgb,I2_gray = readimage(path_infrared,img_infrared,path_visible,img_visible)
    print('【Successfully read image!!】')

    # %%
    # section 2: Resize images based on the minimum imaclosege height
    I1, I2, scale = resizeImage(I1_gray,I2_gray)
    I1_itea = I1

    # %%
    # section 3: Registrate iteratively & Coarse matching

    iterationNum = 1
    iteration = 0
    Runtime = 0
    maxRMSE = 4*np.ceil(I2.shape[0]/300)  # maxRMSE = 12.0
    AffineTrans = np.zeros((3,3,iterationNum))

    while iteration < iterationNum:
        [P1,P2,Rt,corner12,pos_cor1] = hjw_registration(I1,I2,maxtheta=20,maxErr=maxRMSE,iteration=iteration,zoomascend=1,zoomdescend=0,Lc=6,showflag=1,I2ori=I2_gray)
        Runtime += Rt

        I1_itea,affmat = hjw_getAffine(I1_itea,I2,P1,P2)  # % [v1,u1]==[v2,u2]
        AffineTrans[:,:,iteration] = affmat.T
        iteration += 1

    # % Points of I1gray after resize
    P1 = np.hstack(  (  P1 , np.ones((P1.shape[0],1))  )  ) # Matlab中，拼接后P1 是 (98,3) corner12是(2972,2)
    P1 = P1[:,0:2]
    corner12 = corner12[:,0:2]

    # % Correct matches in the source images
    P1[:,1] = I1_gray.shape[0]/2 + scale[0] * (P1[:,1] - I1.shape[0]/2)
    P1[:,0] = I1_gray.shape[1]/2 + scale[0] * (P1[:,0] - I1.shape[1]/2)
    corner12[0:pos_cor1,1] = I1_gray.shape[0]/2 + scale[0] * (corner12[0:pos_cor1,1] - I1.shape[0]/2)
    corner12[0:pos_cor1,0] = I1_gray.shape[1]/2 + scale[0] * (corner12[0:pos_cor1,0] - I1.shape[1]/2)


    P2[:,1] = I2_gray.shape[0]/2 + scale[1] * (P2[:,1] - I2.shape[0]/2)
    P2[:,0] = I2_gray.shape[1]/2 + scale[1] * (P2[:,0] - I2.shape[1]/2)
    corner12[pos_cor1+1:,1] = I2_gray.shape[0]/2 + scale[1] * (corner12[pos_cor1+1:,1] - I2.shape[0]/2)
    corner12[pos_cor1+1:,0] = I2_gray.shape[1]/2 + scale[1] * (corner12[pos_cor1+1:,0] - I2.shape[1]/2)

    # %%
    # section 4: Fine matching
    P3 = hjw_subpixelFine(P1,P2)

    # %%
    # section 5: Show visual registration result
    _,affmat = hjw_getAffine(I1_gray,I2_gray,P1,P3)
    affmat = affmat.T
    print(f'affmat={affmat}')
    
    # ===========================================================================================================
    # ===========================================================================================================
    # I13_python_affmat:
    # [[ 4.65866260e-01 -4.45560274e-03 -2.34593972e-05]
    #  [ 5.39699239e-02  5.23562478e-01  1.57841986e-04]
    #  [ 1.67173712e+02 -4.11006551e+01  1.00000000e+00]]
    # I13_python_correct_dismatchremoval_affmat:
    #[[ 4.65721368e-01 -5.88112539e-03 -2.39554016e-05]
    # [ 5.15559018e-02  5.23629112e-01  1.53931046e-04]
    # [ 1.67641569e+02 -4.08656190e+01  1.00000000e+00]]
    # I13_matlab_affmat:
    # 0.471393843752357	-0.00188117933529857	-9.53282918852061e-06
    # 0.0461492908977536	0.522927251861718	0.000139936847492703
    # 167.840076383476	-42.4535804261491	1



    # I8_python_affmat:
    # [[ 1.20690173e+00  5.51252560e-04 -2.43693714e-05]
    #  [ 6.15683852e-04  1.23961136e+00  3.10564727e-05]
    #  [-4.03688651e+01 -9.78501870e+01  1.00000000e+00]]
    # I8_python_correct_dismatchremoval_affmat:
    # [[ 1.23436831e+00  1.52287937e-02  9.98481516e-07]
    #  [ 9.47062649e-03  1.27818353e+00  5.93842397e-05]
    #  [-4.47167635e+01 -1.10076189e+02  1.00000000e+00]]
    # I8_matlab_affmat:
    # 1.24155304007150	0.00124195690530627	2.38398443270224e-07
    # 0.00490881955508114	1.25799190774813	5.02042307506865e-05
    # -46.6135770449656	-101.818398459276	1


    # I1_python_affmat:
    #[[ 1.17348262e-01 -1.08596530e+00 -5.68946532e-06]
    # [ 9.95731781e-01  3.13281083e-02 -2.65918603e-04]
    # [-1.65277591e+02  7.48647445e+02  1.00000000e+00]]
    # I1_matlab_affmat:
    # 0.215002590003858	-1.11842096989889	0.000176619778079327
    # 1.18794391272921	0.146980130563059	-5.85348472960980e-05
    # -235.076526538735	772.081695043171	1
    # ===========================================================================================================
    # ===========================================================================================================



    Imosaic = hjw_graymosaic(I1_gray,I2_gray,affmat)
    rgb_Imosaic = hjw_rgbmosaic(I1_rgb,I2_rgb,affmat)

    plt.imshow(rgb_Imosaic)
    plt.show()

    