
# 基本库函数
import time
import numpy as np
import cv2
from math import pi,sin,cos
import matplotlib.pyplot as plt
import imutils



# 自定义函数
from hjw_cornerDetection import cornerDetection
from hjw_match import hjw_match
from hjw_showMatch import hjw_showMatch
from hjw_atan import hjw_atan
from hjw_mismatchRemoval import hjw_mismatchRemoval
from multiprocess_hjw_descriptor import multiprocess_hjw_descriptor


# 图像配准主函数
def hjw_registration(I1,I2,maxtheta,maxErr,iteration,zoomascend,zoomdescend,Lc,showflag,I2ori):
    #                                                    1          0
    # %% Corner detection and orientation computed

    cornerdetection_start_time = time.time()
    [cor1,orientation1] = cornerDetection(I1,C=1.5,T_angle=170,sig=3,H=80,L=30,Endpoint=1,Gap_size=1,maxlength=Lc,rflag=iteration==0)
    [cor2,orientation2] = cornerDetection(I2,C=1.5,T_angle=170,sig=3,H=80,L=30,Endpoint=1,Gap_size=1,maxlength=Lc,rflag=iteration==0)
    orientation1 = np.array(orientation1)
    orientation2 = np.array(orientation2)

    cornerdetection_end_time = time.time()
    print('The cornerdetection time is {:.2f}s '.format(cornerdetection_end_time-cornerdetection_start_time))

    cor1 = np.array(cor1)   
    cor2 = np.array(cor2)

    # ===========================================================================================================
    # ===========================================================================================================
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/2021_12_5_2150_I13_cor1.txt',cor1.astype(np.float))
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/2021_12_5_2150_I13_cor2.txt',cor2.astype(np.float))
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/2021_12_5_2150_I13_o1.txt',orientation1.astype(np.float))
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/2021_12_5_2150_I13_o2.txt',orientation2.astype(np.float))
    # # ===========================================================================================================
    # ===========================================================================================================

    pos_cor1 = cor1.shape[0]
    cor12 = np.hstack(    (    np.reshape(cor1[:,1],(-1,1))   ,   np.reshape(cor1[:,0],(-1,1)   )  )   ) 
    cor12 = np.vstack((cor12,np.array([[0,0]])))
    temp = np.hstack(   (   np.reshape(cor2[:,1],(-1,1)) , np.reshape(cor2[:,0],(-1,1))  )  )
    cor12 = np.vstack((cor12,temp))
    
    I1 = I1.astype('float32')
    I2 = I2.astype('float32')
    I2ori = I2ori.astype('float32')

    # %% 
    # show orientation of each corner
    if iteration == 0 and showflag:
        plt.subplot(121)
        plt.imshow(I1,cmap='gray')
        plt.plot(cor1[:,1],cor1[:,0],'y+')
        # ############################################################################### 破案了，scale的值越小，箭头越长
        plt.quiver(cor1[:,1],cor1[:,0],\
                    np.cos(orientation1),-np.sin(orientation1),\
                    color = 'r',scale = 25) 
        plt.title('corner of infrared image')

        plt.subplot(122)
        plt.imshow(I2,cmap='gray')
        plt.plot(cor2[:,1],cor2[:,0],'y+')
        plt.quiver(cor2[:,1],cor2[:,0],\
                    np.cos(orientation2),-np.sin(orientation2),\
                    color = 'r',scale = 25)   
        plt.title('corner of visible image')
        plt.show()

    # %%
    # extract decriptor of Image 1
    descriptor_start_time = time.time()
    print('【【【【 Start computing the descriptor: 】】】】')
    scale12 = 36 # % square width is 6(pixels) * 4(squares) = 24 pixels totally
    
    cols1 = cor1[:,0]         # % swap row and col
    cols1 = np.reshape(cols1,(1,-1)) # % row is x axis | col is y axis
    rows1 = cor1[:,1]
    rows1 = np.reshape(rows1,(1,-1)) 
    
    s1 = scale12*np.ones((cols1.shape))
    if iteration > 0: # % no need to calculate orientation
        o1 = np.zeros(cols1.shape)
    else:
        o1 = np.reshape(orientation1,(1,-1))

    key1 = np.vstack((rows1,cols1,s1,o1))
    des1 = multiprocess_hjw_descriptor(I1,key1)


    # %%
    # % extract descriptors of Image 2 under multi scales
    cols2 = cor2[:,0]  # % swap row and col
    cols2 = np.reshape(cols2,(1,-1))
    rows2 = cor2[:,1]  # % row is x axis | col is y axis
    rows2 = np.reshape(rows2,(1,-1))
    if iteration > 0:
        o2 = np.zeros(cols2.shape)
    else:
        o2 = np.reshape(orientation2,(1,-1))
    
    s2 = scale12 * np.ones(cols2.shape)
    key2 = np.vstack((rows2,cols2,s2,o2))
    
    # % zoom transformation parameters
    zoomstepup = 1
    zoomstepdown = 0.5
    des2 = np.zeros((128,cor2.shape[0],zoomascend+zoomdescend+1))
    des2[:,:,0] = multiprocess_hjw_descriptor(I2,key2)

    level = 0
    scale = I2ori.shape[0] / I2.shape[0]
    print(f'scale = {scale}')
    if iteration == 0:
        for i in range(1,zoomascend+1):
            level += 1
            key2zoom = key2
            I2zoom = cv2.resize(I2ori,None,fx = (1+zoomstepup*i)/scale ,fy = (1+zoomstepup*i)/scale)
            key2zoom[0:2,:] = np.floor(key2zoom[0:2,:] * (1+zoomstepup*i))
            key2zoom[0:3,:] = key2zoom[0:3,:]
            des2[:,:,level] = multiprocess_hjw_descriptor(I2zoom,key2zoom)

        for i in range(1,zoomdescend+1):
            level += 1
            key2zoom = key2
            I2zoom = cv2.resize(I2ori,None,fx = (1-zoomstepdown*i)/scale,fy = (1-zoomstepdown*i)/scale)
            key2zoom[0:2,:] = np.floor(key2zoom[0:2,:] * (1-zoomstepdown*i))
            key2zoom[0:3,:] = key2zoom[0:3,:]
            des2[:,:,level] = multiprocess_hjw_descriptor(I2zoom,key2zoom)


    descriptor_end_time = time.time()
    print('Time of descriptor is {:.2f}'.format(descriptor_end_time-descriptor_start_time))
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I3_des1.txt',des1)
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I3_des2_1.txt',des2[:,:,0])
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I3_des2_2.txt',des2[:,:,1])


    # %%
    # %% *----Matchings the keys from two images---*
    # %% coarsely match with BBF method

    # des1x = np.loadtxt('D:/Neural Network/Sun_liguo/middle_data/I8_des1.txt')
    # des2_1 = np.loadtxt('D:/Neural Network/Sun_liguo/middle_data/I8_des2_1.txt')
    # des2_2 = np.loadtxt('D:/Neural Network/Sun_liguo/middle_data/I8_des2_2.txt')
    # des2x = np.dstack((des2_1,des2_2))
    matchindex1,matchindex2,zoom = hjw_match(des1.T,des2,0.97)
    zoomscale = (zoom == 0)*1 + (0<zoom and zoom <= zoomascend)*(1+zoom*zoomstepup)+ (zoom > zoomascend)* (1-(zoom-zoomascend)*zoomstepdown)
    print(f'zoomscale = {zoomscale}')   

    # ################################################### 注意，下标是先1再0，说明已经为了画图改变顺序了
    regis_points111 = np.hstack((np.reshape(cor1[matchindex1,1],(-1,1)),np.reshape(cor1[matchindex1,0],(-1,1)),np.reshape(o1[0,matchindex1],(-1,1))))   #  % (u,v,orientation)
    regis_points222 = np.hstack((np.reshape(cor2[matchindex2,1],(-1,1)),np.reshape(cor2[matchindex2,0],(-1,1)),np.reshape(o2[0,matchindex2],(-1,1))))
    print(f'Number of features1: {regis_points111.shape[0]} || Number of features2: {regis_points222.shape[0]} ')
    
    # ===========================================================================================================
    # ===========================================================================================================
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I13_python_regispoints111.txt',regis_points111)
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I13_python_regispoints222.txt',regis_points222)
    # ===========================================================================================================
    # ===========================================================================================================
    if regis_points111.shape[0] < 2:
        raise ValueError('Error: No sufficient points(<2) ! Failed to registrate!')
    
    if showflag:
        hjw_showMatch(I1,cv2.resize(I2ori,None,fx = zoomscale/scale,fy = zoomscale/scale),\
                    regis_points111[:,0:2],zoomscale * regis_points222[:,0:2],[],'Putative matches')

    # %% 
    # Scale invariant tilt angle consistency matching

    if iteration == 0:
        delta0 = np.round(180/pi*(regis_points222[:,2]-regis_points111[:,2])) % 360
        dd = 5
        d_delta = np.arange(0,361,dd)
        n_delta,_ = np.histogram(delta0,d_delta)
        n0 = np.argmax(n_delta)
        nmat1 = np.reshape(np.array([(n0+1)**2,n0+1,1]),(1,-1))
        nmat2 = np.reshape(np.array([n0**2,n0,1]),(1,-1))
        nmat3 = np.reshape(np.array([(n0+2)**2,n0+2,1]),(1,-1))
        nmat = np.vstack((nmat1,nmat2,nmat3))


        temp = np.reshape(np.array([n_delta[n0],n_delta[int(n0-1+360/dd*(n0==0))],n_delta[int(n0+1-((n0+1)==360/dd))]]),(3,1))
        nmat = np.dot(np.linalg.inv(nmat),temp).flatten()
        Modetheta_discrete = -nmat[1]/2/nmat[0] # % -b/2a
        Modetheta = Modetheta_discrete * dd # % 抛物线插值

    else:
        Modetheta = 0
    print(f'Modetheta = {Modetheta}')  
    # % image to the same direction then those correct matches line will be parallel

    temp1 = np.hstack(   (   np.reshape(regis_points222[:,0] - I2.shape[1]/2,(-1,1))    ,\
                    np.reshape(regis_points222[:,1] - I2.shape[0]/2,(-1,1)    )   ))

    temp  = Modetheta*pi/180
    temp2 = np.vstack((np.array([cos(temp),-sin(temp)]),np.array([sin(temp),cos(temp)])))

    trans222 = np.dot(temp1,temp2)
    trans222[:,0] += I2.shape[1]/2
    trans222[:,1] += I2.shape[0]/2
    phi_uv = hjw_atan(dy = zoomscale*trans222[:,1]-regis_points111[:,1],\
                    dx = zoomscale*trans222[:,0]+I1.shape[1]-regis_points111[:,0])



    # %% 
    # show result of bilateral matching
    if showflag:
        hjw_showMatch(I1 = I1, I2 = cv2.resize(imutils.rotate(I2,Modetheta),None,fx = zoomscale, fy = zoomscale),\
                    loc1 = regis_points111[:,0:2],\
                    loc2 = zoomscale * trans222[:,0:2],
                    correctPos=[],
                    tname=f'After {iteration} resized and rotated ')


    dd = 5
    d_phi = np.arange(-90,91,5)
    n_phi,_ = np.histogram(phi_uv,bins = d_phi)
    n0 = np.argmax(n_phi)
    interval_index = np.where(n_phi<0.2*n_phi[n0])[0]    
    interval_index -= n0

    left_phi = np.where(interval_index<0)[0]
    right_phi = np.where(interval_index>0)[0]

    maxtheta1 = -dd * interval_index[left_phi[-1]]
    maxtheta2 = dd * interval_index[right_phi[0]]

    temp1 = np.vstack( ( np.array([(n0+1)**2,n0+1,1]),\
                        np.array([(n0+180/dd*(n0==0))**2,n0+180/dd*(n0==0),1]),\
                        np.array([(n0+2-(n0==(180/dd-1)))**2,n0+2-(n0==(180/dd-1)),1]))     )
    temp2 = np.array([n_phi[n0],n_phi[int(n0+180/dd*(n0==0))],n_phi[int(n0+2-(n0==(180/dd-1)))]])
    temp2 = np.reshape(temp2,(3,1))
    nmat = np.dot(np.linalg.inv(temp1),temp2)
    ModePhi_discrete = -nmat[1]/2/nmat[0] - 90/dd   # % -b/2a
    ModePhi = ModePhi_discrete * dd
    delta1 = ModePhi - maxtheta1
    delta2 = ModePhi + maxtheta2

    valid0 = np.where( (phi_uv >= delta1) * (phi_uv <= delta2) > 0 )[0]

    temp = np.hstack((np.reshape(zoomscale*trans222[valid0,1]-regis_points111[valid0,1],(-1,1)),\
                    np.reshape(zoomscale*trans222[valid0,0]+I1.shape[1]-regis_points111[valid0,0],(-1,1))))
    Dist = np.linalg.norm(temp,ord=2,axis = 1)
    meandist = np.mean(Dist)

    valid1 = np.where(  (Dist>=0.75*meandist)  *  (Dist<=1.25*zoomscale*meandist)   >0 )[0]
    regis_points11 = regis_points111[valid0[valid1],:]
    regis_points22 = regis_points222[valid0[valid1],:]

    # ===========================================================================================================
    # ===========================================================================================================
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I13_python_regispoints11.txt',regis_points11)
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/I13_python_regispoints22.txt',regis_points22)
    # ===========================================================================================================
    # ===========================================================================================================

    # %% 
    # show result of Tile angle consistency matching
    if showflag:
        hjw_showMatch(I1 = I1,\
                    I2 = cv2.resize(imutils.rotate(I2,Modetheta),None,fx=zoomscale,fy = zoomscale),\
                    loc1 = regis_points111[valid0[valid1],0:2],\
                    loc2 = zoomscale*trans222[valid0[valid1],0:2],\
                    correctPos=[],\
                    tname=f'After {iteration} Tile angle consistency')

    correctindex = hjw_mismatchRemoval(regis_points11,regis_points22,I1,I2,maxErr)
    regis_points1 = regis_points11[np.array(correctindex),0:2]
    regis_points2 = regis_points22[np.array(correctindex),0:2]

    if showflag:
        hjw_showMatch(I1,I2,regis_points1,regis_points2,[],'Fine matching result')


    Fine_matching_end_time = time.time()
    Runtime = Fine_matching_end_time-cornerdetection_start_time


    # ===========================================================================================================
    # ===========================================================================================================
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/correct_mismatchRemoval_I13_python_regispoints1.txt',regis_points1)
    # np.savetxt('D:/Neural Network/Sun_liguo/middle_data/correct_mismatchRemoval_I13_python_regispoints2.txt',regis_points2)
    # ===========================================================================================================
    # ===========================================================================================================
    print(f'【1】 Completed image registration ')
    return regis_points1,regis_points2,Runtime,cor12,pos_cor1