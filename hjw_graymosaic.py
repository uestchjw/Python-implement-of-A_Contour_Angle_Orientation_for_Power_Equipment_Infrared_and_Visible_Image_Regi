


from scipy import io
import numpy as np
from scipy.interpolate import griddata
import cv2
import matplotlib.pyplot as plt

# def fun(x,y):
#     return x**2+y**2

# xy = np.random.rand(10,2)*5
# z = fun(xy[:,0],xy[:,1])



# znew = griddata(xy,z,(grid_x,grid_y))
# print(znew)
# print(znew.shape)


# # python如何画三维图
# import mpl_toolkits.mplot3d
# ax = plt.subplot(111,projection = '3d')
# ax.plot_surface(xnew,ynew,znew)
# plt.show()

path_infrared = 'D:/Neural Network/Sun_liguo/Example_Images/infrared_images/I13.jpg'
path_visible = 'D:/Neural Network/Sun_liguo/Example_Images/visible_images/V13.jpg'
I1 = cv2.imread(path_infrared,0)
I2 = cv2.imread(path_visible,0)
extend = max(I1.shape)


M = np.array([[0.459843740477301,	-0.00971970394861337,	-4.09401422931936e-05],\
[0.0498730194649677,	0.523588343950058,	0.000158393188726247],\
[169.137019015027,	-40.5830348423779,	1]])
'''破案了 Matlab的.T不是转置的意思，而是取出T属性'''


Imosaic = np.zeros((2*extend+I2.shape[0],2*extend+I2.shape[1]))
v,u = np.mgrid[0:Imosaic.shape[0],0:Imosaic.shape[1]]
u,v = u.flatten('F'),v.flatten('F')  # 这里原来没'F'，我说结果为啥不对呢
u -= extend # (4989696,)
v -= extend
# np.savetxt('D:/u.txt',u,fmt='%d')
# np.savetxt('D:/v.txt',v,fmt='%d')

utvt = np.hstack(  ( np.reshape(u,(-1,1)),np.reshape(v,(-1,1)),np.ones((len(u),1)) )  )
'''经验证，utvt和MATLAB一致'''



utvt = np.dot(utvt,np.linalg.inv(M))
ut = utvt[:,0]/utvt[:,2]
vt = utvt[:,1]/utvt[:,2]
# np.savetxt('D:/ut.txt',ut,fmt='%.4f') 
# np.savetxt('D:/vt.txt',vt,fmt='%.4f') 




# Matlab重排时，列优先
# python重排时，行优先  b = np.reshape(a,(3,2),order='C') 改为 b = np.reshape(a,(3,2),order='F')即可

'''经过验证，utu和vtv，和MATLAB差距在-8之内,可以认为相等'''
M_utu = io.loadmat('D:/M_utu.mat')['utu']
M_vtv = io.loadmat('D:/M_vtv.mat')['vtv']
utu = np.reshape(ut,Imosaic.shape,'F')   
vtv = np.reshape(vt,Imosaic.shape,'F')

# print(utu[1990,750:760])
# print(M_utu[1990,750:760])
# print(np.min(utu-M_utu['utu']))
# print(np.min(vtv-M_vtv['vtv']))

# np.savetxt('D:/utu.txt',utu,fmt='%.4f')
# np.savetxt('D:/vtv.txt',vtv,fmt='%.4f')



'''
为下面的插值函数griddata做准备
xy是原图像I1的二维点对
[[0,0]
 [0,1]
 [0,2]
 ...
 [767,574]
 [767,575]]
xy长度为 768*576 = 442368
'''
I1 = I1.astype(np.float)
x,y = np.mgrid[0:I1.shape[0],0:I1.shape[1]]
x,y = x.flatten(),y.flatten()
xy = [[x[i],y[i]] for i in range(len(x))]
xy = np.array(xy)


'''python中Iterp的图像，像是重叠了好几张图像的结果，再检查一下utu和vtv吧'''
Iterp = griddata(xy,I1.flatten('F'),(utu,vtv),method='cubic')
# Iterp = Iterp.astype('uint8')
# nan是<class 'numpy.float64'>
plt.imshow(Iterp,cmap='gray')
plt.show()
[ix,iy] = np.where(Iterp>=0)
print(ix)
vmin,vmax = min(ix),max(ix)
umin,umax = min(iy),max(iy)
print(vmin,vmax)
print(umin,umax)
Imosaic[extend:extend+I2.shape[0], extend:extend+I2.shape[1] ] = I2



# 遇到透视变换后图像全黑的，需要把M转置一下
# a = cv2.warpPerspective(I1,affmat.T,( 2*extend+I2.shape[1], 2*extend+I2.shape[0] ))
a = cv2.warpPerspective(I1,M.T,(I1.shape[1],I1.shape[0]))
plt.imshow(a,cmap='gray')
plt.show()






def hjw_graymosaic(I1,I2,affmat):
    '''
    input: a pair of source images
    output: a mosaic image based on liniear interplation transformation
    '''

    r1,c1 = I1.shape[0],I1.shape[1]
    r2,c2 = I2.shape[0],I2.shape[1]
    Imosaic = np.zeros(  (r2+2*max(r1,c1), c2+2*max(r1,c1))  )
    # Imosaic:(2336,2136)


    # %%
    affinemat = affmat.T
    leng = np.arange(1,Imosaic.shape[0]+1).tolist()
    v = np.array(leng * Imosaic.shape[1])
    u =[]
    for i in range(Imosaic.shape[1]):
        u.extend([i+1]*Imosaic.shape[0])
    u = np.array(u)
    v -= max(r1,c1)
    u -= max(r1,c1)

    u,v = np.reshape(u,(-1,1)),np.reshape(v,(-1,1))
    # Matlab   A/B  是A乘以B的逆
    utvt = np.dot(    np.hstack(  (u,v,np.ones((v.shape[0],1)))  ),\
                    np.linalg.inv(affinemat) )
    ut = utvt[:,0]/utvt[:,2]
    vt = utvt[:,1]/utvt[:,2]

    utu = np.reshape(ut,(r2+2*max(r1,c1),c2+2*max(r1,c1))  )
    vtv = np.reshape(vt,(r2+2*max(r1,c1),c2+2*max(r1,c1))  )
    print(f'utu.shape={utu.shape}')
    print(f'vtv.shape={vtv.shape}')
    # reshape后，utu = (2336,2136)
    # Iterp = interp2(double(I1), utu, vtv);
    # 插值后，Iterp在Matlab =  (2336,2136)
    interpfun = interpolate.interp2d( np.arange(I1.shape[1]) , np.arange(I1.shape[0]), I1.astype('float32') )
    Iterp = interpfun(np.arange(I1.shape[0]) , np.arange(I1.shape[1]))

    print(f'Iterp.shape={Iterp.shape}')  # Interp = (576,768)      而I1_gray = (768,576)
    return Imosaic