
import numpy as np
import matplotlib.pyplot as plt

def hjw_showMatch(I1,I2,loc1,loc2,correctPos,tname):
    cols = I1.shape[1]
    if I1.shape[0] < I2.shape[0]:
        I1_p  = np.vstack((I1,np.zeros(shape=(I2.shape[0]-I1.shape[0],I1.shape[1]))))
        im3 = np.hstack((I1_p,I2))
    elif I1.shape[0] > I2.shape[0]:
        I2_p = np.vstack((I2,np.zeros(shape=(I1.shape[0]-I2.shape[0],I2.shape[1]))))
        im3 = np.hstack((I1,I2_p))
    else:
        im3 = np.hstack((I1,I2))

    
    plt.imshow(im3,cmap='gray')
    plt.title(tname)
    if len(correctPos) != 0:
        plt.Line2D(xdata = np.hstack((loc1[correctPos,0],loc2[correctPos,0]+cols)).T,\
                    ydata = np.hstack((loc1[correctPos,1],loc2[correctPos,1])).T, \
                    color = 'y', linewidth=1.2)
        plt.plot(loc1[correctPos,0],loc1[correctPos,2],'r+')
        plt.plot(loc2[correctPos,0]+cols,loc2[correctPos,1],'go')
        loc1 = np.delete(loc1,correctPos,axis=0)
        loc2 = np.delete(loc2,correctPos,axis=0)
        plt.Line2D(xdata=np.hstack((loc1[:,0],loc2[:,0]+cols)).T,\
                    ydata=np.hstack((loc1[:,1],loc2[:,1])).T,\
                    color = 'r',linewidth=1.2)
        plt.plot(loc1[:,0],loc1[:,1],'r+')
        plt.plot(loc2[:,0]+cols,loc2[:,1],'ro')
    else:
        plt.plot([loc1[:,0],loc2[:,0]+cols],[loc1[:,1],loc2[:,1]],color = 'y')
        plt.plot(loc1[:,0],loc1[:,1],'r+')
        plt.plot(loc2[:,0]+cols,loc2[:,1],'go')

    plt.show()


