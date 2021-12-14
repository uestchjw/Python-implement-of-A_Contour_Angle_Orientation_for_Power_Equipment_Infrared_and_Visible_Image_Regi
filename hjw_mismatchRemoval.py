
import cv2
import numpy as np
import math
from numpy import random


def hjw_mismatchRemoval(p1,p2,edg1,edg2,maxErr):
    # mismatch removal algorithm
    # % Input: cp_mismatchRemoval(p1,p2,edge1,edge2,maxErr,iteration)
    # % p1: coarse match of image 1
    # % p2: coarse match of image 2

    iteration = 1
    zoomflag = 1
    if p1.shape[0] == 3:
        correctIndex = np.array([1,2,3])
        return correctIndex
    if p1.shape[0] == 2:
        correctIndex = np.array([1,2])
        return correctIndex
    minArea1 = edg1.shape[0]*edg1.shape[1]/80
    minArea2 = edg2.shape[0]*edg2.shape[1]/80

    leng = p1.shape[0]
    eor = np.zeros((1,7))
    # ################################################### 到时候记得把eor第1行去掉


    # %%
    # RANSAC algorithm
    for n in range(min(500,int(leng*(leng-1)*(leng-2)/6))):
        ijk = random.choice(leng,3,replace=False) 
        ir = ijk[0]
        jr = ijk[1]
        kr = ijk[2]

        pp1 = np.vstack(  (np.reshape(p1[ir,0:2],(-1,2)),np.reshape(p1[jr,0:2],(-1,2)),np.reshape(p1[kr,0:2],(-1,2))  )  )
        pp2 = np.vstack(  (np.reshape(p2[ir,0:2],(-1,2)),np.reshape(p2[jr,0:2],(-1,2)),np.reshape(p2[kr,0:2],(-1,2))  )  )
        A1 = p1[ir,:] - p1[jr,:]
        B1 = p1[ir,:] - p1[kr,:]
        A2 = p2[ir,:] - p2[jr,:]
        B2 = p2[ir,:] - p2[kr,:]
        if abs(  (A1[0]*B1[1]-A1[1]*B1[0])  )<minArea1 or abs( (A2[0]*B2[1]-A2[1]*B2[0]) ) < minArea2:
            continue
        ransacERR = getEdgeOverlappingRatio(pp1,pp2,p1[:,0:2],p2[:,0:2],ir,jr,kr,maxErr)
        temp = np.hstack( ( np.array([ir,jr,kr]),ransacERR  ) )
        eor = np.vstack((eor,temp))

    
    eor = eor[1:,:]
    if eor.shape[0] < 3:
        raise Exception('No sufficent matches! ( less than 3 matches obtained ) ')
    ind = np.argmax(eor[:,4])
    base1 = p1[ eor[ind,0:3].astype(int), 0:2 ]
    base2 = p2[ eor[ind,0:3].astype(int), 0:2 ]
    affmat0 = cv2.getAffineTransform(base1.astype('float32'),base2.astype('float32'))
    correctIndex = eor[ind,0:3].astype(int).tolist()

    for i in range(leng):
        if i == eor[ind,0] or i == eor[ind,1] or i == eor[ind,2]:
            continue
        temp = np.reshape(np.hstack((p1[i,0:2],[1])),(1,-1))
        pp1_aff = np.dot(temp,affmat0.T)
        pp2_to_pp1aff = np.abs(p2[i,0:2] - pp1_aff[0,0:2])
        if (pp2_to_pp1aff[0] < maxErr/1.5 and pp2_to_pp1aff[1]<1.5*maxErr) or \
            (pp2_to_pp1aff[0]<1.5*maxErr and pp2_to_pp1aff[1]<maxErr/1.5):
            correctIndex.append(i)
            
    return correctIndex


def getEdgeOverlappingRatio(pp1,pp2,p1,p2,a,b,c,maxErr):
    pp1 = pp1.astype('float32')
    pp2 = pp2.astype('float32')
    affmat = cv2.getAffineTransform(pp1,pp2)

    p1_aff = np.dot(   np.hstack((p1,np.ones((p1.shape[0],1)))) , affmat.T )
    p2_to_p1aff = (p2 - p1_aff[:,0:2])**2

    temp = math.sqrt (np.sum(p2_to_p1aff.flatten())/p2.shape[0])
    pixelDistance = p2_to_p1aff[:,0] + p2_to_p1aff[:,1]

    pixelDistance[[a-1,b-1,c-1]] = 2*maxErr**2  # % ignore origin three points
    u = np.where(pixelDistance < maxErr**2)[0]     # % mount of internal points of fitting result
    minerr = np.min(pixelDistance)
    ind0 = np.argmin(pixelDistance)

    RMSE = np.array([temp ,u.shape[0],minerr,ind0 ])

    return RMSE