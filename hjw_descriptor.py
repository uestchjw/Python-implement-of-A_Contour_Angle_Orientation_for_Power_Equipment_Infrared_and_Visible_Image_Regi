



import numpy as np
# import time
from math import atan, exp
from math import floor, pi,sqrt
# 对应MATLAB中的imfilter滤波
from scipy.ndimage import correlate
from numba import jit



# ############################################################################### I1是double型；key1是numpy，shape = (4,855)

def hjw_descriptor(img1,keypoints):
    img = img1/255
    NBP = 4
    NBO = 8
    key_num = keypoints.shape[1]
    descriptors = np.ones((NBP*NBP*NBO,1))
    # descriptors = np.zeros((NBP*NBP*NBO,key_num))


    # %%
    # %======= compute the image gradient vertors =====%
    
    # 这段代码运行是很快的，只需要0.03s

    # % M is the height of image, N is the width of image
    M = img.shape[0]
    N = img.shape[1]
    du_filter = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    dv_filter = np.array([[0,-1,0],[0,0,0],[0,1,0]])
    duv_filter = np.diag(np.array([-1,0,1]))  # 主对角线
    dvu_filter = np.fliplr(duv_filter)        # 副对角线

    gradient_u = correlate(img,du_filter)
    gradient_v = correlate(img,dv_filter)
    gradient_uv = correlate(img,duv_filter)
    gradient_vu = correlate(img,dvu_filter)
    gradient_x = 1.414*gradient_u+gradient_uv-gradient_vu
    gradient_y = 1.414*gradient_v+gradient_uv+gradient_vu

    magnitudes = np.hypot(gradient_x,gradient_y)

    # %%
    # %====== compute angle of gradient =======%

    # 这里用时7s

    angles = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            if gradient_x[i,j] == 0:
                if gradient_y[i,j] > 0 :
                    angles[i,j] = 3*pi/2
                elif gradient_y[i,j] == 0:
                    angles[i,j] = 100
                    # 用100代替Matlab中值为NaN,即0/0
                else:
                    angles[i,j] = pi/2
            else:
                if gradient_x[i,j] > 0:
                    if gradient_y[i,j]>=0:
                        angles[i,j] = atan(gradient_y[i,j]/gradient_x[i,j])
                    else:
                        angles[i,j] = atan(gradient_y[i,j]/gradient_x[i,j])+2*pi
                else:
                    angles[i,j] = atan(gradient_y[i,j]/gradient_x[i,j])+pi

    x = keypoints[0,:].astype(np.int)  # % u
    y = keypoints[1,:].astype(np.int)  # % v
    s = keypoints[2,:].astype(np.int)  # % scale
    theta = keypoints[3,:]
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # 等价于[xx,yy] = meshgrid(-NBP/2:NBP/2)
    xx,yy = np.meshgrid(np.arange(-NBP/2,NBP/2+1),np.arange(-NBP/2,NBP/2+1))
    # wincoef = math.exp(-(xx**2+yy**2)/NBP**2*2)
    wincoef = -(xx**2+yy**2)/NBP**2*2
    for p in range(wincoef.shape[0]):
        for q in range(wincoef.shape[1]):
            wincoef[p,q] = exp(wincoef[p,q])

    temp = int(NBP/2)


    # ##### 感觉下面可以写成一个函数，再用numba加速
    # descriptors = hjw_numba(key_num,magnitudes,s,x,y,sintheta,costheta,NBP,NBO,M,N,angles,theta,temp,wincoef)
    # return descriptors




    # %%
    # %========= compute descriptors ==========%
    ############################################################################### key_num = 855
    for p in range(key_num):
        magnitude = magnitudes
        sp = s[p]
        xp = x[p]   # % u == x
        yp = y[p]   # % v == y
        sinth0 = sintheta[p]
        costh0 = costheta[p]
        W = sp      # % scale
        ss = W/2    # % scale/2
        descriptor = np.zeros((NBP,NBP,NBO))  # % 4 * 4 * 8

        # ############################################################################### matlab中，pp是73*60的矩阵
        # ############################################################################### python中，magnitudes是(768,576)的numpy

        # pp = magnitudes[int(max(-W,1-yp)+yp-1):int(min(+W,M-yp)+yp),int(max(-W,1-xp)+xp-1):int(min(W,N-xp)+xp)]
        pp = magnitudes[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp]

        pp = pp/np.max(pp)
        # 在MATLAB中，xx是4380*1,从小到大
        # 在python中，xx是 (2701)
        xx = np.sort(pp.flatten())
        xind1 = round((1-1/5)*xx.shape[0])
        xind2 = round((1-2/5)*xx.shape[0])
        xind3 = round((1-3/5)*xx.shape[0])
        xind4 = round((1-4/5)*xx.shape[0])

        # pp = (pp>=xx[xind1-1]).astype('float')+((pp<xx[xind1-1]) * (pp>=xx[xind2-1])).astype('float')*0.75+ ((pp<xx[xind2-1]) * (pp>=xx[xind3-1])).astype('float')*0.5 + ((pp<xx[xind3-1]) * (pp>=xx[xind4])).astype('float')*0.25
        pp = (pp>=xx[xind1-1])+(  (pp<xx[xind1-1])  * (pp>=xx[xind2-1]) )*0.75+ (    (pp<xx[xind2-1])  * (pp>=xx[xind3-1])   )*0.5 + (  (pp<xx[xind3-1]) * (pp>=xx[xind4-1])   )*0.25
        magnitude[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp] = pp

        dx,dy = np.meshgrid(np.arange(max(-W,1-xp),min(W,N-xp)+1),np.arange(max(-W,1-yp),min(+W,M-yp)+1))
        nx = (costh0*dx+sinth0*dy) / ss
        ny = (-sinth0*dx+costh0*dy) / ss

        
        # ############################################################################### matlab中，nx，ny是73*60的矩阵，dx,dy都是73*60的矩阵
        # ############################################################################### 这里python和MATLAB有个区别，
        # ############################################################################### 就是MATLAB中a是二维矩阵，a[1]也是一个数
        # ############################################################################### python中a是二维numpy，a[1]就是一个vector
        # ############################################################################### matlab中列优先存储
        
        dx,dy,nx,ny = dx.flatten('F').astype('int'),dy.flatten('F').astype('int'),nx.flatten('F').astype('float32'),ny.flatten('F').astype('float32')


        # 主要的计算时间在下面这个循环里面
        # 等价于 for kk = 1:numel(dx)
        for kk in range(dx.shape[0]):  # 2701

            mag = magnitude[yp+dy[kk]-1,xp+dx[kk]-1]
            angle = ( angles[yp+dy[kk]-1,xp+dx[kk]-1] - theta[p]) % pi
            # nt = NBO*angle/pi
            binx = floor(nx[kk]-0.5)
            biny = floor(ny[kk]-0.5)
            # bint = floor(nt)
            bint = floor(NBO*angle/pi)
            rbinx = nx[kk] - (binx+0.5)
            rbiny = ny[kk] - (biny+0.5)
            rbint = NBO*angle/pi - bint

            for dbinx in range(2):
                for dbiny in range(2):
                    for dbint in range(2):
                        if  angles[yp+dy[kk]-1,xp+dx[kk]-1] == 100 or binx+dbinx < -temp or binx+dbinx >= temp or biny+dbiny < -temp or biny+dbiny >= temp :
                            continue
                        else:
                           descriptor[binx+dbinx+temp,biny+dbiny+temp,(bint+dbint) % NBO] += wincoef[binx+dbinx+temp,biny+dbiny+temp] * mag * abs(1-dbinx-rbinx) * abs(1-dbiny-rbiny) * abs(1-dbint-rbint)

        descriptors = np.hstack(   (  descriptors,   np.reshape( descriptor.ravel('F')/np.linalg.norm(descriptor),(-1,1) )    )   )
    # 因为一开始创建的初始值是descriptors = np.ones((128,1)) 因此第一列不能要
    descriptors = descriptors[:,1:]
    return descriptors







# @jit (nopython=True)
# def hjw_numba(key_num,magnitudes,s,x,y,sintheta,costheta,NBP,NBO,M,N,angles,theta,temp,wincoef):
#     descriptors = np.ones((NBP*NBP*NBO,1))
#     for p in range(key_num):
#         magnitude = magnitudes
#         sp = s[p]
#         xp = x[p]   # % u == x
#         yp = y[p]   # % v == y
#         sinth0 = sintheta[p]
#         costh0 = costheta[p]
#         W = sp      # % scale
#         ss = W/2    # % scale/2
#         descriptor = np.zeros((NBP,NBP,NBO))  # % 4 * 4 * 8

#         # ############################################################################### matlab中，pp是73*60的矩阵
#         # ############################################################################### python中，magnitudes是(768,576)的numpy

#         # pp = magnitudes[int(max(-W,1-yp)+yp-1):int(min(+W,M-yp)+yp),int(max(-W,1-xp)+xp-1):int(min(W,N-xp)+xp)]
#         pp = magnitudes[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp]

#         pp = pp/np.max(pp)
#         # 在MATLAB中，xx是4380*1,从小到大
#         # 在python中，xx是 (2701)
#         xx = np.sort(pp.flatten())
#         xind1 = round((1-1/5)*xx.shape[0])
#         xind2 = round((1-2/5)*xx.shape[0])
#         xind3 = round((1-3/5)*xx.shape[0])
#         xind4 = round((1-4/5)*xx.shape[0])

#         # pp = (pp>=xx[xind1-1]).astype('float')+((pp<xx[xind1-1]) * (pp>=xx[xind2-1])).astype('float')*0.75+ ((pp<xx[xind2-1]) * (pp>=xx[xind3-1])).astype('float')*0.5 + ((pp<xx[xind3-1]) * (pp>=xx[xind4])).astype('float')*0.25
#         pp = (pp>=xx[xind1-1])+(  (pp<xx[xind1-1])  * (pp>=xx[xind2-1]) )*0.75+ (    (pp<xx[xind2-1])  * (pp>=xx[xind3-1])   )*0.5 + (  (pp<xx[xind3-1]) * (pp>=xx[xind4-1])   )*0.25
#         magnitude[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp] = pp



#         # dx,dy = np.meshgrid(np.arange(max(-W,1-xp),min(W,N-xp)+1),np.arange(max(-W,1-yp),min(+W,M-yp)+1))
#         hjwx = np.arange(max(-W,1-xp),min(W,N-xp)+1)
#         hjwy = np.arange(max(-W,1-yp),min(+W,M-yp)+1)

#         dx = np.zeros((1,len(hjwx)))
#         for ipp in range(len(hjwy)):
#             dx = np.vstack(    (   dx,   np.reshape(hjwx,(1,-1))    )  )
#         dx = dx[1:,:]
        
#         dy = np.zeros((len(hjwy),1))
#         for ipp in range(len(hjwx)):
#             dy = np.hstack(  (dy,np.reshape(hjwy,(-1,1)))  )
#         dy = dy[:,1:]

#         nx = (costh0*dx+sinth0*dy) / ss
#         ny = (-sinth0*dx+costh0*dy) / ss

        
#         # ############################################################################### matlab中，nx，ny是73*60的矩阵，dx,dy都是73*60的矩阵
#         # ############################################################################### 这里python和MATLAB有个区别，
#         # ############################################################################### 就是MATLAB中a是二维矩阵，a[1]也是一个数
#         # ############################################################################### python中a是二维numpy，a[1]就是一个vector
#         # ############################################################################### matlab中列优先存储
#         # res_dx = 
#         # for iip in dx.shape[1]:

#         res = np.array([0])
#         print(' ')
#         for i in range(dx.shape[1]):
#             res = np.hstack((res,dx[:,i]))
#         dx = res[1:,]

#         res = np.array([0])
#         for i in range(dy.shape[1]):
#             res = np.hstack((res,dy[:,i]))
#         dy = res[1:,]
#         res = np.array([0])
#         for i in range(nx.shape[1]):
#             res = np.hstack((res,nx[:,i]))
#         nx = res[1:,]
#         res = np.array([0])
#         for i in range(ny.shape[1]):
#             res = np.hstack((res,ny[:,i]))
#         ny = res[1:,]

#         # dx,dy,nx,ny = dx.flatten('F'),dy.flatten('F'),nx.flatten('F'),ny.flatten('F')
#         dx,dy = dx.astype('int'),dy.astype('int')
#         nx,ny = nx.astype('float32'),ny.astype('float32')
#         dx,dy,nx,ny = dx.flatten('F').astype('int'),dy.flatten('F').astype('int'),nx.flatten('F').astype('float32'),ny.flatten('F').astype('float32')


#         # 主要的计算时间在下面这个循环里面
#         # 等价于 for kk = 1:numel(dx)
#         for kk in range(dx.shape[0]):  # 2701

#             mag = magnitude[yp+dy[kk]-1,xp+dx[kk]-1]
#             angle = ( angles[yp+dy[kk]-1,xp+dx[kk]-1] - theta[p]) % pi
#             # nt = NBO*angle/pi
#             binx = floor(nx[kk]-0.5)
#             biny = floor(ny[kk]-0.5)
#             # bint = floor(nt)
#             bint = floor(NBO*angle/pi)
#             rbinx = nx[kk] - (binx+0.5)
#             rbiny = ny[kk] - (biny+0.5)
#             rbint = NBO*angle/pi - bint

#             for dbinx in range(2):
#                 for dbiny in range(2):
#                     for dbint in range(2):
#                         if  angles[yp+dy[kk]-1,xp+dx[kk]-1] == 100 or binx+dbinx < -temp or binx+dbinx >= temp or biny+dbiny < -temp or biny+dbiny >= temp :
#                             continue
#                         else:
#                            descriptor[binx+dbinx+temp,biny+dbiny+temp,(bint+dbint) % NBO] += wincoef[binx+dbinx+temp,biny+dbiny+temp] * mag * abs(1-dbinx-rbinx) * abs(1-dbiny-rbiny) * abs(1-dbint-rbint)

#         descriptors = np.hstack(   (  descriptors,   np.reshape( descriptor.ravel('F')/np.linalg.norm(descriptor),(-1,1) )    )   )
#     # 因为一开始创建的初始值是descriptors = np.ones((128,1)) 因此第一列不能要
#     descriptors = descriptors[:,1:]

#     return descriptors




