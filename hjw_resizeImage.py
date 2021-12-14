


import cv2

# 将图像调整为以最小高度为基准，图像比例不变

# .shape()得到的是(行数，列数)，但是在resize时，要给定(列数，行数)
def resizeImage(infrared_gray,visible_gray):
    print('【Images are Resized！】')
    height1 = infrared_gray.shape[0]
    height2 = visible_gray.shape[0]
    min_height = min(height1,height2)

    scale1 = height1/min_height
    scale2 = height2/min_height
    scale = [scale1, scale2]
    new_infrared_gray = cv2.resize(infrared_gray,(int(infrared_gray.shape[1]/scale1),min_height))
    new_visible_gray = cv2.resize(visible_gray,(int(visible_gray.shape[1]/scale2),min_height))
    
    return new_infrared_gray,new_visible_gray,scale