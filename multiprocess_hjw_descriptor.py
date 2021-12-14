



import numpy as np
from math import atan, exp
from math import floor, pi
from numba import jit
from scipy.ndimage import correlate # 对应MATLAB中的imfilter滤波

def multiprocess_hjw_descriptor(img1,keypoints):

    @jit(nopython=True)
    def fun(num,magnitude,yp,dy,xp,dx,angles,theta,p,nx,ny,NBO,temp,wincoef):
        descriptor = np.zeros((NBP,NBP,NBO))
        for kk in range(num):
            mag = magnitude[yp+dy[kk]-1,xp+dx[kk]-1]
            angle = ( angles[yp+dy[kk]-1,xp+dx[kk]-1] - theta[p]) % pi
            binx = floor(nx[kk]-0.5)
            biny = floor(ny[kk]-0.5)
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
                            descriptor[binx+dbinx+temp,biny+dbiny+temp,(bint+dbint) % NBO] += wincoef[binx+dbinx+temp,biny+dbiny+temp] \
                                                                                                * mag * abs(1-dbinx-rbinx) * abs(1-dbiny-rbiny) \
                                                                                                * abs(1-dbint-rbint)
        return descriptor
    img = img1/255
    NBP = 4
    NBO = 8
    key_num = keypoints.shape[1]
    descriptors = np.ones((NBP*NBP*NBO,1))


    # %%
    # %======= compute the image gradient vertors =====%
    
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

    xx,yy = np.meshgrid(np.arange(-NBP/2,NBP/2+1),np.arange(-NBP/2,NBP/2+1))
    wincoef = -(xx**2+yy**2)/NBP**2*2
    for p in range(wincoef.shape[0]):
        for q in range(wincoef.shape[1]):
            wincoef[p,q] = exp(wincoef[p,q])

    temp = int(NBP/2)

    # =======================================================================================================================
    # =======================================================================================================================
    # 尽量避免类内属性访问
    zeros = np.zeros # 不能加括号
    arange = np.arange
    meshgrid = np.meshgrid
    # =======================================================================================================================
    # =======================================================================================================================
    # %%
    # %========= compute descriptors ==========%
    for p in range(key_num):
        magnitude = magnitudes
        sp = s[p]
        xp = x[p]   # % u == x
        yp = y[p]   # % v == y
        sinth0 = sintheta[p]
        costh0 = costheta[p]
        W = sp      # % scale
        ss = W/2    # % scale/2
        pp = magnitudes[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp]

        pp = pp/np.max(pp)
        xx = np.sort(pp.flatten())
        xind1 = round((1-1/5)*xx.shape[0])
        xind2 = round((1-2/5)*xx.shape[0])
        xind3 = round((1-3/5)*xx.shape[0])
        xind4 = round((1-4/5)*xx.shape[0])

        pp = (pp>=xx[xind1-1])+(  (pp<xx[xind1-1])  * (pp>=xx[xind2-1]) )*0.75+ (    (pp<xx[xind2-1])  * (pp>=xx[xind3-1])   )*0.5 + (  (pp<xx[xind3-1]) * (pp>=xx[xind4-1])   )*0.25
        magnitude[max(-W,1-yp)+yp-1:min(+W,M-yp)+yp,max(-W,1-xp)+xp-1:min(W,N-xp)+xp] = pp

        dx,dy = meshgrid(arange(max(-W,1-xp),min(W,N-xp)+1),arange(max(-W,1-yp),min(+W,M-yp)+1))
        nx = (costh0*dx+sinth0*dy) / ss
        ny = (-sinth0*dx+costh0*dy) / ss
        dx,dy,nx,ny = dx.flatten('F').astype('int'),dy.flatten('F').astype('int'),nx.flatten('F').astype('float32'),ny.flatten('F').astype('float32')
        
        descriptor1 = fun(dx.shape[0],magnitude,yp,dy,xp,dx,angles,theta,p,nx,ny,NBO,temp,wincoef)
        descriptors = np.hstack(   (  descriptors,   np.reshape( descriptor1.ravel('F')/np.linalg.norm(descriptor1),(-1,1) )    )   )
    # 因为一开始创建的初始值是descriptors = np.ones((128,1)) 因此第一列不能要
    descriptors = descriptors[:,1:]

    return descriptors









