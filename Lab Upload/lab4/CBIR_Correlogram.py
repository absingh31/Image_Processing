import pickle
import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2


class CBIR_Correlogram:
    """Autocorrelogram based CBIR

     Content Based Image Retrieval using Color Autocorrelogram
    """

    def __init__(self, database='image/', use_index=None, save_index=False):
        self.database = database

        # index the database or load previously indexed database
        if save_index is False: 
            self.index = self.indexor(save_index)
        else:
            print('Loading indexed database...')
            with open(use_index, 'rb') as indx:
                self.index = indx
        return None

    def _display(self, cv_image, title='Image'):
        """Convert OpenCV images from BGR to RGB for Pyplot"""
        plt_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(plt_image)
        plt.show()
        return True

    def _manhattan(self, x, y):
        """
        Cityblock or Manhattan Distance between two lists
        """
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d

    def unique(self, a):
        """
        remove duplicates from input list
        """
        order = np.lexsort(a.T)
        a = a[order]
        diff = np.diff(a, axis = 0)
        ui = np.ones(len(a), 'bool')
        ui[1:] = (diff != 0).any(axis = 1)
        return a[ui]
    
    def isValid(self, X, Y, point):
        """
        Check if point is a valid pixel
        """
        if point[0] < 0 or point[0] >= X:
            return False
        if point[1] < 0 or point[1] >= Y:
            return False
        return True

    def getNeighbors(self, X, Y, x, y, dist):
        """
        Find pixel neighbors according to various distances
        """
        cn1 = (x + dist, y + dist)
        cn2 = (x + dist, y)
        cn3 = (x + dist, y - dist)
        cn4 = (x, y - dist)
        cn5 = (x - dist, y - dist)
        cn6 = (x - dist, y)
        cn7 = (x - dist, y + dist)
        cn8 = (x, y + dist)
     
        points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
        Cn = []
     
        for i in points:
            if self.isValid(X, Y, i):
              Cn.append(i)

        return Cn

    def correlogram(self, photo, Cm, K):
        """
        Get auto correlogram
        """
        X, Y, t = photo.shape
        colorsPercent = []

        for k in K:
            # print "k: ", k
            countColor = 0
            color = []
            for i in Cm:
               color.append(0)
     
            for x in range(0, X, int(round(X / 10))):
                for y in range(0, Y, int(round(Y / 10))):

                    Ci = photo[x][y]
                    Cn = self.getNeighbors(X, Y, x, y, k)
                    for j in Cn:
                        Cj = photo[j[0]][j[1]]
     
                        for m in range(len(Cm)):
                            if np.array_equal(Cm[m], Ci) and np.array_equal(Cm[m], Cj):
                                countColor = countColor + 1
                                color[m] = color[m] + 1

            for i in range(len(color)):
                color[i] = float(color[i]) / countColor
            
            colorsPercent.append(color)
        return colorsPercent


    def query(self, query, rank=10, precision_recall=False, relevant=None):
        """Performs an image query in the CBIR database"""

        # load the query image and show it
        print("Processing Query Image...")
        query_image= cv2.imread(query)
        self._display(cv_image=query_image, title='Query')

        # describe the query in the same way that we did in
        # index.py -- a 3D RGB histogram with 8 bins per channel
        query_features = self.autoCorrelogram(query_image)

        print("Searching...")
        results = self.search(query_features)

        print("Displaying results...")
        # loop over the top ranks
        for j in range(rank):
            # grab the result (we are using row-major order) and
            # load the result image
            (score, imageName) = results[j]
            path = self.database + "/%s" % (imageName)
            result = cv2.imread(path)
            self._display(result,
                          title='Result#{} Path:{} Score:{:.2f}'.format(j+1, path, score))

        if precision_recall == True and relevant is not None:
            result_images = [i[1] for i in results]
            relevant_ranks = sorted([result_images.index(x)+1 for x in relevant])
            num_relevant = range(len(relevant))
            precision = np.divide(num_relevant, relevant_ranks)
            recall = np.divide(num_relevant, len(relevant))
            # plot precision-recall curve
            plt.plot(recall, precision, 'r.-')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Graph')
            plt.axis([0, 1, 0, 1.05])
            plt.show()

        return None

    def autoCorrelogram(self, image, K=64):
        """
        The functions for computing color correlogram. 
        To improve the performance, we consider to utilize 
        color quantization to reduce image into 64 colors. 
        So the K value of k-means should be 64.

        image:
         The numpy ndarray that describe an image in 3 channels.
        K:
         K neighbours for color quantization to reduce image
         into fewer colors
        """
        Z = image.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))

        # according to "Image Indexing Using Color Correlograms" paper
        K = [i for i in range(1, 9, 2)]

        colors64 = self.unique(np.array(res))

        result = self.correlogram(res2, colors64, K)
        return result

    def indexor(self, save_index):
        # initialize the index dictionary to store our our quantifed
        # images, with the 'key' of the dictionary being the image
        # filename and the 'value' our computed features
        index = {}
        print('Indexing the Image Database...')
        # use glob to grab the image paths and loop over them
        for imagePath in glob.glob(self.database + "/*.*"):
            # extract our unique image ID (i.e. the filename)
            k = imagePath[imagePath.rfind("/") + 1:]
            print("Processing {}".format(k))

            # load the image, describe it using our RGB histogram
            # descriptor, and update the index
            image = cv2.imread(imagePath)
            features = self.autoCorrelogram(image)
            index[k] = features
        if save_index:
            # we are now done indexing our image -- now we can write our
            # index to disk
            print("Saving indexed database...")
            with open('index', 'wb') as f:
                f.write(pickle.dumps(index))
        return index

    def search(self, query_features):
        # initialize our dictionary of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the features
            # in our index and our query features -- using the
            # chi-squared distance which is normally used in the
            # computer vision field to compare histograms
            d = self._manhattan(features, query_features)

            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[k] = d

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our results
        return results
