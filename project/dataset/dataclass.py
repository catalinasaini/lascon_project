import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog

class MNIST:
    """
        The MNIST class serves a base class to generate the MSNIT dataset to be inputed to the network.  
    """
    
    def __init__(self):
        """
            MNIST Instantiation.
            
            From a Google Colab, the mnist dataset was download from:
            
            self.mnist = tf.keras.datasets.mnist # 28 x 28 pixels
            (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        """
        
        # Load raw files
        self.x_train = np.load('dataset/x_train_data.npy')
        self.y_train = np.load('dataset/y_train_data.npy')
        self.x_test = np.load('dataset/x_test_data.npy')
        self.y_test = np.load('dataset/y_test_data.npy')
        
        # Load feature vectors
        self.x_train_feat_vect = np.load('dataset/x_train_feat_vect.npy')
        self.x_test_feat_vect = np.load('dataset/x_test_feat_vect.npy')    
    
    def hog_transformation(self, image):
        """
            Corrected function for HOG transformation with specific stride.

        Args:
            image (_type_): _description_
            stride (int, optional): _description_. Defaults to 7.

        Returns:
            _type_: _description_
        """
        # Declare variables
        stride = 7
        feature_vectors = []

        # Size of the image, cells, and blocks
        height, width = image.shape
        cell_size = 14

        # Iterate over the image with the given stride
        for y in range(0, height - cell_size + 1, stride):
            for x in range(0, width - cell_size + 1, stride):
                cell_region = image[y:y+cell_size, x:x+cell_size]

                # Calculate HOG features for the cell
                fd, hog_image = hog(
                    cell_region,
                    orientations=9,
                    pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(1, 1),
                    visualize=True,
                    feature_vector=True
                )

                feature_vectors.append(fd)

        # Concatenate all feature vectors
        feature_vector = np.concatenate(feature_vectors)

        # print("Feature vector:", feature_vector)
        # print("Feature vector length:", len(feature_vector))
        # print("Feature descriptor:", fd)
        # print("Feature descriptor length:", len(fd))

        return feature_vector, hog_image
    
    def bin_feature_values(self, feature_vector):
        '''
        Binning the feature values into four categories
        '''

        binned_vector = []

        for value in feature_vector:
            if value < 0.25:
                binned_vector.append([1, 0, 0, 0])  # UltraLow
            elif value < 0.50:
                binned_vector.append([0, 1, 0, 0])  # MediumLow
            elif value < 0.75:
                binned_vector.append([0, 0, 1, 0])  # MediumHigh
            else:
                binned_vector.append([0, 0, 0, 1])  # High

        return np.array(binned_vector).flatten()    # flattening preserves properties but in 1-dimension array
    
    def plot_images(self, original_image, hog_image):
        """
        Function to plot the original image and its HOG representation side by side.

        Args:
            original_image (_type_): _description_
            hog_image (_type_): _description_
        """
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Image')

        plt.show()
        
    def process_and_store(self, x):
        '''
        Process and store binned features vectors for training data
        '''

        binned_features = []

        for image in x:
            feature_vector, hog_image = self.hog_transformation(image) # Apply HOG transformation to each image

            binned_feature = self.bin_feature_values(feature_vector)   # Bin the feature vector values

            binned_features.append(binned_feature)                # Store the binned feature vector

            # plot_images(image, hog_image)                       # Plot original and HOG transformed images

        binned_features = np.array(binned_features)               # Convert list to numpy array

        return binned_features

        feature_x_train = process_and_store(x_train).copy()
        feature_x_test = process_and_store(x_test).copy()

        # print(feature_x_train)
        # print(feature_x_test)

        # plt.show()




     

        