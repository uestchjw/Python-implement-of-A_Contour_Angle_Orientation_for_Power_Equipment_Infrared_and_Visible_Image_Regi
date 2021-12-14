



import cv2
import numpy as np
def hjw_getAffine(I1,I2,p1,p2):
    p1 = p1.astype('float32')
    p2 = p2.astype('float32')
    if p1.shape[0] == 3:
        affmat = cv2.getAffineTransform(p1,p2)
        print('     Affine transformation applied!')
        Iaffine = cv2.warpAffine(I1,affmat,I2.shape)
    elif p1.shape[0] >= 4:
        # error: (-215:Assertion failed) src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4 in function 'cv::getPerspectiveTransform'
        affmat,_ = cv2.findHomography(p1,p2,cv2.RANSAC,5.0)
        print('     Projective transformation applied!')
        Iaffine = cv2.warpPerspective(I1,affmat,I2.shape)
        
    # 这里，2个点的先不考虑
    else:
        raise Exception('Transformation Failed! No sufficient Matches!!')
    
    return Iaffine,affmat