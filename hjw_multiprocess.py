

# 多进程运算
from multiprocessing import Pool,Process
import multiprocessing

import numpy as np
import matplotlib
from matplotlib import scale
from numpy.lib.function_base import sinc
from scipy.io import loadmat
import numpy as np
import math
from math import ceil, cos, pi, sin, sqrt
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import correlate,convolve


from hjw_readimage import *
from hjw_resizeImage import *
from hjw_registration import cp_registration
from hjw_cornerDetection import *
from hjw_atan import *
from hjw_descriptor import *
from hjw_test_descriptor import tnd



if __name__ == '__main__':

    # source image path
    path_infrared = 'D:/Neural Network/Sun_liguo/Example_Images/infrared_images/'
    path_visible = 'D:/Neural Network/Sun_liguo/Example_Images/visible_images/'

    img_infrared = 'I1.jpg'
    img_visible = 'V1.jpg'


        # %%
    # section 1: Read source images 
    # 这里省略了张正友畸变校正法，去除红外图片的畸变
    infrared_rgb,infrared_gray,visible_rgb,visible_gray = readimage(path_infrared,img_infrared,path_visible,img_visible)









        # %%
    # section 2: Resize images based on the minimum imaclosege height
    new_infrared_gray, new_visible_gray = resizeImage(infrared_gray,visible_gray)

    I1 = new_infrared_gray
    I2 = new_visible_gray
    I2ori = visible_gray
    ################################################################################ 调整之后的大小都是(768,576),和Matlab一致








    # %%
        # section 3: Registrate iteratively & Coarse matching

    iterationNum = 1
    iteration = 0
    Runtime = 0
    maxRMSE = 4*ceil(new_visible_gray.shape[0]/300.)
    AffineTrans = np.zeros((3,3,iterationNum))


    Lc = 6

    cornerdetection_start_time = time.time()
    [cor1,orientation1] = cornerDetection(I1,C=1.5,T_angle=170,sig=3,H=90,L=50,Endpoint=1,Gap_size=1,maxlength=Lc,rflag=iteration==0)
    [cor2,orientation2] = cornerDetection(I2,C=1.5,T_angle=170,sig=3,H=90,L=50,Endpoint=1,Gap_size=1,maxlength=Lc,rflag=iteration==0)
    cornerdetection_end_time = time.time()
    print('The cornerdetection time is {:.2f}s '.format(cornerdetection_end_time-cornerdetection_start_time))





    cor1 = np.array(cor1)   # cor1.shape = (855,2)   cor2.shape = (1948,2)
    cor2 = np.array(cor2)
    a,b = cor1[:,1].copy(),cor1[:,0].copy()
    a,b = np.reshape(a,(-1,1)),np.reshape(b,(-1,1))
    cor12 = np.hstack((a,b))
    cor12 = np.vstack((cor12,np.array([[0,0]])))
    c,d = cor2[:,1].copy(),cor2[:,0].copy()
    c,d = np.reshape(c,(-1,1)),np.reshape(d,(-1,1))
    temp = np.hstack((c,d))
    cor12 = np.vstack((cor12,temp))

    
    I1 = I1.astype('float32')
    I2 = I2.astype('float32')
    I2ori = I2ori.astype('float32')



    orientation1,orientation2 = np.array(orientation1),np.array(orientation2)
    xu1 = np.cos(orientation1)        
    yv1 = np.sin(orientation1)
    xu2 = np.cos(orientation2)
    yv2 = np.sin(orientation2)
    

    plt.subplot(121)
    plt.imshow(I1,cmap='gray')
    plt.plot(cor1[:,1],cor1[:,0],'y+')
    # ############################################################################### 破案了，scale的值越小，箭头越长
    plt.quiver(cor1[:,1],cor1[:,0],xu1,yv1,color = 'r',scale = 25) 
    plt.title('corner of infrared image')

    plt.subplot(122)
    plt.imshow(I2,cmap='gray')
    plt.plot(cor2[:,1],cor2[:,0],'y+')
    plt.quiver(cor2[:,1],cor2[:,0],xu2,yv2,color = 'r',scale = 25)   
    plt.title('corner of vivible image')
    plt.show()










    descriptor_start_time = time.time()
    print('【【【【 Start compute the descriptor: 】】】】')
    scale12 = 36 # % square width is 6(pixels) * 4(squares) = 24 pixels totally
    
    cols1 = cor1[:,0].copy()         # % swap row and col
    cols1 = np.reshape(cols1,(1,-1)) # % row is x axis | col is y axis
    rows1 = cor1[:,1].copy()
    rows1 = np.reshape(rows1,(1,-1)) 
    
    s1 = scale12*np.ones((cols1.shape))
    if iteration > 0: # % no need to calculate orientation
        o1 = np.zeros(cols1.shape)
    else:
        o1 = np.reshape(np.array(orientation1),(1,-1))

    key1 = np.vstack((rows1,cols1,s1,o1))



    # ############################################################################### 重点是用多进程方法处理这个descriptor
    # NBP = 4
    # NBO = 8
    # key_num = key1.shape[1]
    # descriptors = np.ones((128,1))


    # img = I1
    # keypoints = key1
    # M = img.shape[0]
    # N = img.shape[1]
    # du_filter = np.reshape(np.array([-1,0,1]),(1,-1))
    # dv_filter = du_filter.T
    # du_filter = np.vstack((du_filter,np.zeros((2,3))))
    # dv_filter = np.hstack((dv_filter,np.zeros((3,2))))

    # duv_filter = np.diag(np.array([-1,0,1]))  # 主对角线
    # dvu_filter = np.fliplr(duv_filter)        # 副对角线

    # gradient_u = correlate(img,du_filter)
    # gradient_v = correlate(img,dv_filter)
    # gradient_uv = correlate(img,duv_filter)
    # gradient_vu = correlate(img,dvu_filter)

    # gradient_x = 1.414*gradient_u+gradient_uv-gradient_vu
    # gradient_y = 1.414*gradient_v+gradient_uv+gradient_vu

    # # % dx_filter = [-1 0 1];
    # # % dy_filter = dx_filter';
    # # % gradient_x = imfilter(img, dx_filter);
    # # % gradient_y = imfilter(img, dy_filter);

    # magnitudes = np.hypot(gradient_x,gradient_y)









    # angles = np.zeros((M,N))
    # for i in range(M):
    #     for j in range(N):
    #         if gradient_x[i,j] == 0:
    #             if gradient_y[i,j] > 0 :
    #                 angles[i,j] = pi/2
    #             elif gradient_y[i,j] == 0:
    #                 angles[i,j] = 0
    #             else:
    #                 angles[i,j] = 3*pi/2
    #         else:
    #             if gradient_x[i,j] > 0:
    #                 if gradient_y[i,j]>=0:
    #                     angles[i,j] = math.atan(gradient_y[i,j]/gradient_x[i,j])
    #                 else:
    #                     angles[i,j] = math.atan(gradient_y[i,j]/gradient_x[i,j])+2*pi
    #             else:
    #                 angles[i,j] = math.atan(gradient_y[i,j]/gradient_x[i,j])+pi

    # x = keypoints[0,:]  # % u
    # y = keypoints[1,:]  # % v
    # s = keypoints[2,:]  # % scale
    # theta = keypoints[3,:]
    # sintheta = np.sin(theta)
    # costheta = np.cos(theta)


    # # 等价于[xx,yy] = meshgrid(-NBP/2:NBP/2)
    # xx,yy = np.meshgrid(np.arange(-NBP/2,NBP/2+1),np.arange(-NBP/2,NBP/2+1))
    # # wincoef = math.exp(-(xx**2+yy**2)/NBP**2*2)
    # temp = -(xx**2+yy**2)/NBP**2*2
    # for p in range(temp.shape[0]):
    #     for q in range(temp.shape[1]):
    #         temp[p,q] = math.exp(temp[p,q])
    # wincoef = temp

    key_num = key1.shape[1]
    p = np.arange(key_num)
    with Pool(key_num) as hjw:
        result = hjw.map(tnd,p)




    des1 = hjw_descriptor(I1,key1)


