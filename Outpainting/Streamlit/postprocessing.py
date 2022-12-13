import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def reduce_colors(image,n_colors):
    if image.ndim == 4:
        image = image[0:,:,:,:]
    image = np.clip(image,0,1)
    image = image*255
    image = np.reshape(image,(256*256,3))
    kmeans = KMeans(n_clusters=n_colors,random_state=0).fit(image)
    labels = kmeans.predict(image)
    p = lambda x : kmeans.cluster_centers_[x]



    return p(labels).reshape((256,256,3))
