import pickle
import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2


class HistCBIR:
    """Histogram CBIR

     Content Based Image Retrieval using Color Based Histogram
    """

    def __init__(self, database='imageDB/', bins=[8, 8, 8], save_index=False):
        self.bins = bins
        self.database = database
        # index the database
        self.index = self.indexor(save_index)
        return None

    def _display(self, cv_image, title='Image'):
        """Convert OpenCV images from BGR to RGB for Pyplot"""
        plt_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(plt_image)
        plt.show()
        return True

    def _chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d

    def query(self, query, rank=10, precision_recall=False, relevant=None):
        """Performs an image query in the CBIR database"""

        # load the query image and show it
        print("Processing Query Image...")
        query_image= cv2.imread(query)
        self._display(cv_image=query_image, title='Query')

        # describe the query in the same way that we did in
        # index.py -- a 3D RGB histogram with 8 bins per channel
        query_features = self.descriptor(query_image)

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
            plt.title('Precision-Recall Graph for Bins : {}'.format(self.bins))
            plt.axis([0, 1, 0, 1.05])
            plt.show()

        return None

    def descriptor(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
                            None, self.bins,
                            [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist)
        # return out 3D histogram as a flattened array
        return hist.flatten()

    def indexor(self, save_index):
        # initialize the index dictionary to store our our quantifed
        # images, with the 'key' of the dictionary being the image
        # filename and the 'value' our computed features
        index = {}

        # use glob to grab the image paths and loop over them
        for imagePath in glob.glob(self.database + "/*.*"):
            # extract our unique image ID (i.e. the filename)
            k = imagePath[imagePath.rfind("/") + 1:]

            # load the image, describe it using our RGB histogram
            # descriptor, and update the index
            image = cv2.imread(imagePath)
            features = self.descriptor(image)
            index[k] = features
        if save_index:
            # we are now done indexing our image -- now we can write our
            # index to disk
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
            d = self._chi2_distance(features, query_features)

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