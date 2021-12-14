
import cv2
import numpy as np


def hjw_subpixelFine(P1,P2):
    # %CP_SUBPIXELFINE refine the coarse match coordinates in subpixel level

    leng = P1.shape[0]
    affmat,_ = cv2.findHomography(P1,P2,method=cv2.RANSAC,ransacReprojThreshold=5.0)
    P2pro = np.dot( np.hstack((P1,np.ones((leng,1)))) ,\
                    affmat.T)
    P2pro[:,0] /= P2pro[:,2]
    P2pro[:,1] /= P2pro[:,2]
    P2pro = P2pro[:,0:2]
    devia_P = (P2-P2pro)**2
    # MATLAB中，sum(a,2)返回每一行总和的列向量
    devia_P = np.sqrt(devia_P[:,0]+devia_P[:,1])
    max_Devia = np.max(devia_P)
    iteration = 0
    P2fine = P2
    while max_Devia > 0.05 and iteration < 20:
        iteration += 1
        print(f'\nsubpixel iteration = {iteration}\tmax_Devia = {max_Devia}')
        index = np.argsort(devia_P)
        ind1 = round(1/4*len(index))
        P2fine[index[ind1-1:]-1,:] = P2pro[index[ind1-1:]-1,:]
        attmat = cv2.findHomography(P1,P2fine,cv2.RANSAC,5)
        P2pro = np.dot( np.hstack((P1,np.ones((leng,1)))) ,\
                    affmat.T)
        P2pro[:,0] /= P2pro[:,2]
        P2pro[:,1] /= P2pro[:,2]
        P2pro = P2pro[:,0:2]
        devia_P = (P2fine-P2pro)**2
        devia_P = np.sqrt(devia_P[:,0]+devia_P[:,1])
        max_Devia = np.max(devia_P)
        
    return P2fine