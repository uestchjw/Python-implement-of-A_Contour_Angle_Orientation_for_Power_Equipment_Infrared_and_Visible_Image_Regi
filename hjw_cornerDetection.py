


import cv2
import numpy as np
import math
from math import pi, sqrt
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean


# %       Input :
# %       I -  the input image, it could be gray, color or binary image. If I is
# %           empty([]), input image can be get from a open file dialog box.
# %       C -  denotes the minimum ratio of major axis to minor axis of an ellipse, 
# %           whose vertex could be detected as a corner by proposed detector.  
# %           The default value is 1.5.
# %       T_angle -  denotes the maximum obtuse angle that a corner can have when 
# %           it is detected as a true corner, default value is 162.
# %       Sig -  denotes the standard deviation of the Gaussian filter when
# %           computeing curvature. The default sig is 3.
# %       H,L -  high and low threshold of Canny edge detector. The default value
# %           is 0.35 and 0.
# %       Endpoint -  a flag to control whether add the end points of a curve
# %           as corner, 1 means Yes and 0 means No. The default value is 1.
# %       Gap_size -  a paremeter use to fill the gaps in the contours, the gap
# %           not more than gap_size were filled in this stage. The default 
# %           Gap_size is 1 pixels.



# %       Output:
# %       cout - 
# %       angle - 
# %       edge_map -

def curve_tangent(cur,center):
    # 假定cur的格式为cur = [[1,2],[3,4],[5,6]]
    direction =[]
    for i in range(1,3):
        if i == 1:
            curve = cur[center-1::-1]
        else:
            curve = cur[center-1:]
        
        L = len(curve)
        if L>3:
            if curve[0]!=curve[L-1]:
                M = math.ceil(L/2.)
                x1,y1 = curve[0][0],curve[0][1]
                x2,y2 = curve[M-1][0],curve[M-1][1]
                x3,y3 = curve[L-1][0],curve[L-1][1]
            else:
                M1 = math.ceil(L/3.)
                M2 = math.ceil(2*L/3.)
                x1,y1 = curve[0][0],curve[0][1]
                x2,y2 = curve[M1-1][0],curve[M1-1][1]
                x3,y3 = curve[M2-1][0],curve[M2-1][1]

            if abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2)) < 1e-8:  # $ straight line
                comp = complex(curve[L-1][0]-curve[0][0],curve[L-1][1]-curve[0][1])
                tangent_direction = np.angle(comp)
            else:
                # % Fit a circle
                x0 = 1/2*(-y1*x2**2+y3*x2**2-y3*y1**2-y3*x1**2-y2*y3**2+x3**2*y1+y2*y1**2-y2*x3**2-y2**2*y1+y2*x1**2+y3**2*y1+y2**2*y3)/(-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2)
                y0 = -1/2*(x1**2*x2-x1**2*x3+y1**2*x2-y1**2*x3+x1*x3**2-x1*x2**2-x3**2*x2-y3**2*x2+x3*y2**2+x1*y3**2-x1*y2**2+x3*x2**2)/(-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2)
                # % R = (x0-x1)^2+(y0-y1)^2

                radius_direction = np.angle(complex(x0-x1,y0-y1))
                adjacent_direction = np.angle(complex(x2-x1,y2-y1))
                tangent_direction = np.sign(math.sin(adjacent_direction-radius_direction))*math.pi/2+radius_direction
        
        else: # % very short line
            tangent_direction = np.angle(complex(curve[L-1][0]-curve[0][0],curve[L-1][1]-curve[0][1]))
        direction.append(tangent_direction*180/math.pi)
    
    ang = abs(direction[0]-direction[1])
    return ang





def extract_curve(canny,Gap_size):
    # %   Function to extract curves from binary edge map, if the endpoint of a
    # %   contour is nearly connected to another endpoint, fill the gap and continue
    # %   the extraction. The default gap size is 1 pixles.
    
    # Gap_size的意义就是，在横或纵小于Gap_size的距离内，都可以认为是连续的
    # 这里补充欧式距离和曼哈顿距离（城市街区距离）
    
    [row,column] = canny.shape
    BW1 = np.zeros(shape=(row+2*Gap_size,column+2*Gap_size))
    BW_edge = np.zeros((row,column))
    BW1[Gap_size:Gap_size+row,Gap_size:Gap_size+column] = canny
    [r,c] = np.where(BW1==255)
    cur_num = 0
    curve = {}
    # 提取边缘图中所有的连续轮廓并存为元胞数组
    while r.shape[0] > 0:
        point = [r[0],c[0]]
        cur = [point.copy()]   ########  注意：这里有个copy的问题

        # 这里有实时删除原轮廓点的操作
        BW1[point[0],point[1]] = 0
        # 比如这一点是x行y列，那么就在[x-Gap_size:x+Gap_size, y-Gap_size:y+Gap_size]这个正方形内再搜索curve的下一个点
        # 注意：Matlab下标是从1开始的，并且python的索引是不要最后一个的
        [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)
        while I.shape[0]>0:
            dist = (I-Gap_size)**2+(J-Gap_size)**2
            # 返回最小值的索引
            index = np.where(dist==min(dist))[0]
            # 之所以下一个curve点不直接是[I[index],J[index]],是因为I,J是截出来的小矩块内的索引，不是全局矩阵的索引

            point[0]+=I[index][0]-Gap_size
            point[1]+=J[index][0]-Gap_size # % find continous point
            cur += [point.copy()]  # cur的形式为[[1,2],[3,4]]
            BW1[point[0],point[1]] = 0
            [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)
        


        # 为什么这里2个方向就够了？
        # 可能是因为canny检测的，一个点，在距他曼哈顿距离为1的地方，最多只有2个点存在，所以2个方向就够
        # 然而经过验证，发现在曼哈顿距离为1的地方，at least 存在着3个点的
        # 所以我个人感觉需要三个方向，当然4个方向最好


        # % Extract edge towards another direction
        point = [r[0],c[0]]
        BW1[point[0],point[1]] = 0
        [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)
        while I.shape[0]>0:
            dist = (I-Gap_size)**2+(J-Gap_size)**2
            # 返回最小值的索引
            index = np.where(dist==min(dist))[0]
            # 之所以下一个curve点不直接是[I[index],J[index]],是因为I,J是截出来的小矩块内的索引，不是全局矩阵的索引
            point[0]+=I[index][0]-Gap_size
            point[1]+=J[index][0]-Gap_size # % find continous point
            cur += [point.copy()]
            BW1[point[0],point[1]] = 0
            [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)
        

        point = [r[0],c[0]]
        BW1[point[0],point[1]] = 0
        [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)        
        while I.shape[0]>0:
            dist = (I-Gap_size)**2+(J-Gap_size)**2
            # 返回最小值的索引
            index = np.where(dist==min(dist))[0]
            # 之所以下一个curve点不直接是[I[index],J[index]],是因为I,J是截出来的小矩块内的索引，不是全局矩阵的索引

            point[0]+=I[index][0]-Gap_size
            point[1]+=J[index][0]-Gap_size # % find continous point
            cur += [point.copy()]  # cur的形式为[[1,2],[3,4]]
            BW1[point[0],point[1]] = 0
            [I,J] = np.where(BW1[point[0]-Gap_size:point[0]+Gap_size+1,point[1]-Gap_size:point[1]+Gap_size+1]==255)

        if len(cur) > (canny.shape[0]+canny.shape[1])/60:
            cur_num += 1
            # 从扩展Gap_size  ->  Original image     cur的形式为[[1,2],[3,4]]
            for i in range(len(cur)):
                for j in range(len(cur[i])):
                    cur[i][j] -= Gap_size
            curve[cur_num] = cur
        
        [r,c] = np.where(BW1==255)
    

    # % 将轮廓元胞数组中的起始点存起来并判定是否为闭合轮廓
    curve_start = []
    curve_end = []
    curve_mode = []
    for i in range(cur_num):      
        curve_start.append(curve[i+1][0])  # curve的键是从1开始的
        curve_end.append(curve[i+1][-1])
        if (curve_start[i][0]-curve_end[i][0])**2+(curve_start[i][1]-curve_end[i][1])**2 <= 4:  # % 32
            curve_mode.append('loop')
        else:
            curve_mode.append('line')
        
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  原文这里有个还原边缘图像，我没做 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    return curve,curve_start,curve_end,curve_mode,cur_num



def get_corner(curve,curve_start,curve_end,curve_mode,curve_num,canny,sig,Endpoint,C,T_angle,maxlength,rflag):
    corner_num = 0
    cout = []
    angle = []

    GaussianDie0ff = 0.0001
    pw = np.arange(1,31)
    ssq = sig*sig   # sig = 3
    kk = -(pw*pw)
    kk = kk.astype(np.float)  # 记得变成浮点数

    for i in range(len(kk)):
        kk[i]/=(2*ssq)
        kk[i] = math.exp(kk[i])
    pp = np.where(kk>GaussianDie0ff)
    width = np.max(pp)+1  # width = 12    注意这里要用np.max，不能用max
    
    t = np.arange(-width,width+1)  # 高斯窗
    t = t.astype(np.float)    # int -> float
    gau = -(t*t)
    for i in range(len(gau)):
        gau[i] /= (2*ssq)
        gau[i] = math.exp(gau[i])

    # 归一化
    gau = gau/sum(gau)
    gau = gau.tolist()
    sigmaLs = maxlength # 6 和MATLAB一致

    for i in range(1,curve_num+1):
        x = []
        y = []
        for j in curve[i]:
            x.append(j[0])
            y.append(j[1])
        W = width
        L = len(x)
        if L > W:
            # % Calculate curvature
            if curve_mode[i-1] == 'loop':
                xL = x[L-W:L+1]
                xL.extend(x)
                xL.extend(x[0:W+1])
                yL = y[L-W:L+1]
                yL.extend(y)
                yL.extend(y[0:W+1])
            else:
                kx = [2*x[0]]*W
                px = x[W:0:-1]
                cx = [kx[j]-px[j] for j in range(len(kx))]


                kkx = [2*x[L-1]]*W
                ppx = x[L-2:L-W-2:-1]
                ccx = [kkx[j]-ppx[j] for j in range(len(kkx))]


                xL = cx
                xL.extend(x)
                xL.extend(ccx)   # % 总长度2w+L

                ky = [2*y[0]]*W
                py = y[W:0:-1]
                cy = [ky[j]-py[j] for j in range(len(ky))]

                kky = [2*y[L-1]]*W
                ppy = y[L-2:L-W-2:-1]
                ccy = [kky[j]-ppy[j] for j in range(len(kky))]

                yL = cy
                yL.extend(y)
                yL.extend(ccy)
            
            xx = np.convolve(xL,gau)
            xx = xx[W:L+3*W]
            yy = np.convolve(yL,gau)
            yy = yy[W:L+3*W]

            Xu = [xx[1]-xx[0]]
            kk = xx[2:L+2*W]
            pp = xx[0:L+2*W-2]
            mm = [(kk[j]-pp[j])/2. for j in range(len(kk))]
            Xu.extend(mm)
            nn = [xx[L+2*W-1]-xx[L+2*W-2]]
            Xu.extend(nn)

            Yu = [yy[1]-yy[0]]
            kk = yy[2:L+2*W]
            pp = yy[0:L+2*W-2]
            mm = [(kk[j]-pp[j])/2. for j in range(len(kk))]
            Yu.extend(mm)
            nn = [yy[L+2*W-1]-yy[L+2*W-2]]
            Yu.extend(nn)

            Xuu = [Xu[1]-Xu[0]]
            kk = Xu[2:L+2*W]
            pp = Xu[0:L+2*W-2]
            mm = [(kk[j]-pp[j])/2. for j in range(len(kk))]
            Xuu.extend(mm)
            nn = [Xu[L+2*W-1]-Xu[L+2*W-2]]
            Xuu.extend(nn)

            Yuu = [Yu[1]-Yu[0]]
            kk = Yu[2:L+2*W]
            pp = Yu[0:L+2*W-2]
            mm = [(kk[j]-pp[j])/2. for j in range(len(kk))]
            Yuu.extend(mm)
            nn = [Yu[L+2*W-1]-Xu[L+2*W-2]]
            Yuu.extend(nn)

            aa = [Xu[j]*Yuu[j] for j in range(len(Xu))]
            bb = [Xuu[j]*Yu[j] for j in range(len(Xuu))]
            a = [aa[j]-bb[j] for j in range(len(aa))]

            cc = [Xu[j]*Xu[j] for j in range(len(Xu))]
            dd = [Yu[j]*Yu[j] for j in range(len(Yu))]
            b = [cc[j]+ dd[j] for j in range(len(cc))]
            b = [b[j]**1.5 for j in range(len(b))]
            K = [abs(a[j]/b[j]) for j in range(len(a))]
            K = [math.ceil(K[j]*100)/100 for j in range(len(K))]




            # % Find curvature local maxima as corner candidates
            extremum = []
            N = len(K)
            n = 0
            Search = 1
            for j in range(N-1):
                if (K[j+1]-K[j])*Search > 0:
                    n += 1
                    extremum.extend([j])  # % In extremum, odd points is minima and even points is maxima
                    Search = -Search
            if len(extremum)%2 == 0:      # 保证了extremum的长度为奇数，为下面for k in range(1,n,2):不会导致索引超出范围
                n += 1
                extremum.extend([N])

            n = len(extremum)
            flag = [1]*n
            lambda_LR = []
            # % Compare with adaptive local threshold to remove round corners
            for k in range(1,n,2):
                if extremum[k-1]-2 <= 0:
                    hh = K[extremum[k]-1::-1]
                else:
                    hh = K[extremum[k]-1:extremum[k-1]-2:-1]
                pp = K[extremum[k]-1:extremum[k+1]]

                index1 = hh.index(min(hh))
                index2 = pp.index(min(pp))
                ROS = K[extremum[k]-index1-1:extremum[k]+index2]
                K_thre = C*mean(ROS)                      # C = 1.5
                if K[extremum[k]-1] < K_thre:
                    flag[k] = 0
                else:
                    lambda_LR.append([extremum[k], index1,index2])
            extremum = extremum[1::2]
            flag = flag[1::2]           
            temp = [p for p,x in enumerate(flag) if  x==1]
            kk = []
            for j in temp:
                kk.extend([extremum[j]])
            extremum = kk

            #  % Check corner angle to remove false corners due to boundary noise and trivial details
            smoothed_curve = []
            for p in range(len(xx)):
                smoothed_curve.append([xx[p],yy[p]])
            n = len(extremum)
            flag = [1]*n
            for j in range(n):
                if n == 1:
                    ang = curve_tangent(smoothed_curve[0:L+2*W],extremum[0])
                elif j == 0:
                    ang = curve_tangent(smoothed_curve[0:extremum[j+1]],extremum[j])
                elif j == n-1:
                    ang = curve_tangent(smoothed_curve[extremum[j-1]-1:L+2*W],extremum[j]-extremum[j-1]+1)
                else:
                    ang = curve_tangent(smoothed_curve[extremum[j-1]-1:extremum[j+1]],extremum[j]-extremum[j-1]+1)
                if ang > T_angle and ang < (360-T_angle):
                    flag[j] = 0

            ppp,qqq = [],[]
            temp = [p for p,x in enumerate(flag) if x!=0]
            for p in temp:
                ppp.append(extremum[p])
                qqq.append(lambda_LR[p])    
            extremum = ppp
            lambda_LR = qqq

            for p in range(len(extremum)):
                extremum[p]-=W
            true_corner = [p for p,x in enumerate(extremum) if x>0 and x<=L]

            ppp,qqq = [],[]
            for p in true_corner:
                ppp.append(extremum[p])
                qqq.append(lambda_LR[p])
            extremum = ppp
            lambda_LR = qqq
            n = len(extremum)

            for j in range(n):
                cout.append(curve[i][extremum[j]-1])
                if rflag:
                    # %**********    CAO: adaptive  parameters algorithm  *********% 
                    xcor = curve[i][extremum[j]-1][1]
                    ycor = curve[i][extremum[j]-1][0]
                    retail = len(curve[i])
                    lengthL = min([lambda_LR[j][1],extremum[j]])
                    lengthR = min([lambda_LR[j][2],retail-extremum[j]+1])
                    coefL = -((np.arange(0,lengthL)/lengthL)**2/2)   
                    coefL = np.exp(coefL)
                    coefL /= sum(coefL)
                    coefR = -(np.arange(lengthR-1,-1,-1)/lengthR)**2/2
                    coefR = np.exp(coefR)
                    coefR /= sum(coefR)

                    temp = curve[i].copy()
                    temp = np.array(temp)
                    xL = np.sum(coefL * temp[extremum[j]+1-lengthL-1:extremum[j],1])
                    yL = np.sum(coefL * temp[extremum[j]+1-lengthL-1:extremum[j],0])
                    xR = np.sum(coefR * temp[extremum[j]-1:extremum[j]+lengthR-1,1])
                    yR = np.sum(coefR * temp[extremum[j]-1:extremum[j]+lengthR-1,0])

                    vL = [xL-xcor, yL-ycor]
                    vR = [xR-xcor, yR-ycor]
                    norm1 = sqrt(vL[0]**2+vL[1]**2)
                    norm2 = sqrt(vR[0]**2+vR[1]**2)
                    vm = []

                    if norm1==0 or norm2 == 0:
                            orientation1 = 0
                    else:
                        vm.extend([vL[0]/norm1+vR[0]/norm2])    
                        vm.extend([vL[1]/norm1+vR[1]/norm2])
                    # %*************   Assign Main Orientation  **********%
                        deltax = vm[0]
                        deltay = vm[1]
                        if deltax == 0:
                            if deltay == 0:
                                orientation1 = 0
                            elif deltay > 0:
                                orientation1 = pi/2
                            else:
                                orientation1 = 3*pi/2
                        else:
                            if deltay >= 0 and deltax > 0: # % quadrant I
                                orientation1 = math.atan(deltay / deltax)
                            elif deltay < 0 and deltax >0: # % quadrant IV
                                orientation1 = math.atan(deltay / deltax)+2*pi
                            elif (deltay >= 0 and deltax<0) or (deltay<0 and deltax<0): # % quadrant II & III
                                orientation1 = math.atan(deltay / deltax)+pi
                    angle.extend([orientation1])

    if Endpoint:
        for i in range(1,curve_num+1):
            retail = len(curve[i])  
            if retail > 0 and curve_mode[i-1] == 'line':
                # % Start point compare with detected corners
                temp = cout.copy()
                temp = np.array(temp)
                kk = np.array(curve_start[i-1]*temp.shape[0])
                kk = np.reshape(kk,(-1,2))
                compare_corner = temp-kk
                compare_corner = compare_corner**2
                compare_corner = compare_corner[:,0]+compare_corner[:,1]
                if min(compare_corner) > 25:  # % Add end points far from detected corners
                    cout.append(curve_start[i-1])

                    if rflag:
                        xx = curve[i][0][1]
                        yy = curve[i][0][0]
                        temp = min(retail,maxlength)
                        coef2 = -((np.arange(temp-1,-1,-1)/sigmaLs)**2/2.)
                        coef2 = np.exp(coef2)
                        coef2 /= sum(coef2)

                        temp1 = curve[i].copy()
                        temp1 = np.array(temp1)

                        xL = sum(coef2 * temp1[0:min(retail,maxlength),1])
                        yL = sum(coef2 * temp1[0:min(retail,maxlength),0])
                        deltax = xL-xx
                        deltay = yL-yy
                        if deltax == 0:
                            if deltay == 0:
                                orientation2 = 0
                            elif deltay > 0:
                                orientation2 = pi/2
                            else:
                                orientation2 = 3*pi/2
                        else:
                            if deltay >= 0 and deltax > 0: # % quadrant I
                                orientation2 = math.atan(deltay / deltax)
                            elif deltay < 0 and deltax >0: # % quadrant IV
                                orientation2 = math.atan(deltay / deltax)+2*pi
                            elif (deltay >= 0 and deltax<0) or (deltay<0 and deltax<0): # % quadrant II & III
                                orientation2 = math.atan(deltay / deltax)+pi
                        angle.extend([orientation2]) 
                # % End point compare with detected corners
                temp = cout.copy()
                temp = np.array(temp)
                kk = np.array(curve_end[i-1]*temp.shape[0])
                kk = np.reshape(kk,(-1,2))
                compare_corner = temp-kk
                compare_corner = compare_corner**2
                compare_corner = compare_corner[:,0]+compare_corner[:,1]
                if min(compare_corner)>25 :
                    cout.append(curve_end[i-1])
                    if rflag:
                        xx = curve[i][-1][1]
                        yy = curve[i][-1][0]
                        
                        coef1 = -((np.arange(0,retail-max(1,retail-maxlength+1)+1)/sigmaLs)**2/2.)
                        coef1 = np.exp(coef1)
                        coef1 /= sum(coef1)

                        temp1 = curve[i].copy()
                        temp1 = np.array(temp1)

                        xL = sum(coef1 * temp1[max(1,retail-maxlength+1)-1:retail,1])
                        yL = sum(coef1 * temp1[max(1,retail-maxlength+1)-1:retail,0])
                        deltax = xL-xx
                        deltay = yL-yy
                        if deltax == 0:
                            if deltay == 0:
                                orientation3 = 0
                            elif deltay > 0:
                                orientation3 = pi/2
                            else:
                                orientation3 = 3*pi/2
                        else:
                            if deltay >= 0 and deltax > 0: # % quadrant I
                                orientation3 = math.atan(deltay / deltax)
                            elif deltay < 0 and deltax >0: # % quadrant IV
                                orientation3 = math.atan(deltay / deltax)+2*pi
                            elif (deltay >= 0 and deltax<0) or (deltay<0 and deltax<0): # % quadrant II & III
                                orientation3 = math.atan(deltay / deltax)+pi
                        angle.extend([orientation3])
    
    return cout,angle




def cornerDetection(I,C,T_angle,sig,H,L,Endpoint,Gap_size,maxlength,rflag):
    
    canny = cv2.Canny(I,L,H)   # python-opencv的Canny，得到的是0和255，不是0和1
    plt.imshow(canny,cmap='gray')
    plt.show()
    curve,curve_start,curve_end,curve_mode,cur_num = extract_curve(canny,Gap_size)
    # Detect corners
    cout,angle = get_corner(curve,curve_start,curve_end,curve_mode,cur_num,canny,sig,Endpoint,C,T_angle,maxlength,rflag)
    return cout,angle