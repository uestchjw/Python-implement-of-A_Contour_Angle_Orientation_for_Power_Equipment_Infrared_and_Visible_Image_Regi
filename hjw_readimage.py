
import cv2

def readimage(path_infrared,img_infrared,path_visible,img_visible):
    infrared_rgb = cv2.imread(path_infrared+img_infrared,1)
    infrared_gray = cv2.imread(path_infrared+img_infrared,0)
    visible_rgb = cv2.imread(path_visible+img_visible,1)
    visible_gray = cv2.imread(path_visible+img_visible,0)
    
    return infrared_rgb,infrared_gray,visible_rgb,visible_gray
