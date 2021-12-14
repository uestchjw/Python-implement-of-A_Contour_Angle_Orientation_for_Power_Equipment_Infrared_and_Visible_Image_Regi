import numpy as np


def hjw_match(des1,des2,distRatio):
    # %%
    zoomvote = np.zeros((3,des2.shape[2]))
    match1 = np.zeros((des1.shape[0],des2.shape[2]))
    match2 = np.zeros((des2.shape[1],des2.shape[2]))

    # % For each descriptor in the first image, select its match to second image.

    for i in range(des1.shape[0]):
        for j in range(des2.shape[2]):
            des2t = des2[:,:,j]

            dotprods = np.dot(  np.reshape(des1[i,:],(1,-1)) ,  des2t  )
            # dotprods = np.dot(des1[i,:],des2t)  # % Computes vector of dot products
            vals1 = np.sort(np.arccos(dotprods)) 
            indx1 = np.argsort(np.arccos(dotprods))  # % Take inverse cosine and sort results

            # % Check if nearest neighbor has angle less than distRatio times 2nd.
            zoomvote[0,j] = 10  #  % > 2*pi is OK
            if vals1[0][0] < vals1[0][1]*distRatio:
                zoomvote[0,j] = vals1[0][0]
                match1[i,j] = indx1[0][0] # % store the matches of each zoom time
        
        mintheta = np.min(zoomvote[0,:])
        ind = np.argmin(zoomvote[0,:])
        if mintheta != 10:
            zoomvote[1,ind] += 1


    # %%
    des1t = des1.T
    for j in range(des2.shape[2]):
        des2zoom = des2[:,:,j].T
        for i in range(des2zoom.shape[0]):
            dotprods = np.dot( np.reshape(des2zoom[i,:],(1,-1)) , des1t )
            # dotprods = np.dot(des2zoom[i,:],des1t)
            vals1 = np.sort(np.arccos(dotprods))
            indx1 = np.argsort(np.arccos(dotprods))
            if vals1[0][0] < distRatio * vals1[0][1]:
                match2[i,j] = indx1[0][0]


    # %% 
    # bilateral match
    zoomvote[2,:] = des1.shape[0] * np.ones((1,des2.shape[2]))  # %assume all of match1 are correct
    for j in range(des2.shape[2]):
        for i in range(des1.shape[0]):
            if match1[i,j] > 0:
                if match2[int(match1[i,j]),j] != i:
                    match1[i,j] = 0
                    zoomvote[2,j] -= 1
            else:
                zoomvote[2,j] -= 1
    
    maxvote = np.max(zoomvote[2,:])
    zoom = np.argmax(zoomvote[2,:])


    # %% 
    # obtain match index
    correctindex1 = []
    correctindex2 = []
    for i in range(des1.shape[0]):
        if match1[i,zoom] > 0:
            correctindex1.append(i)
            correctindex2.append(int(match1[i,zoom]))

    return correctindex1,correctindex2,zoom