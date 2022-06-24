import random as rd
import tensorflow as tf
import numpy as np
import os

# GET NUMPY NDARRAY!!!!! #

#-----------------------------------------------------------------------------------------------------------------------------#

# path_input_siamese = "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\"
# path_conditions = ['Loose','Misalign','Unbalance']

# mypath = "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\Loose\\"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# for files in onlyfiles:
#     path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\Loose\\{files}"


# from os import listdir
# from os.path import isfile, join
# mypath = "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\Loose\\"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# from matplotlib.image import imread
# import matplotlib.pyplot as plt

# img_list = []

# for files in onlyfiles:
#     path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\Loose\\{files}"
#     img = imread(path)
#     img_list.append(img)

# # plt.imshow(img_list[0])
# # plt.show()

# print(img_list[0])

# from PIL import Image
# img = Image.open("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half\\Loose\\Loose03per.png").convert('L')
# img = np.asarray(img)/255

# if tf.config.list_physical_devices('GPU'):
#     print('using')
# else:
#     print('not')

# def batch_label():
    
#     half50_labels = []
#     half100_labels = [] 
#     similar_labels = []

#     for batch in half50:
#         for label in batch[1]:
#             half50_labels.append(label.numpy())

#     for batch in half100:
#         for label in batch[1]:
#             half100_labels.append(label.numpy())

#     for index, label in enumerate(half50_labels):
#         if label == half100_labels[index]:
#             similar_labels.append(float(1))     
#         else:
#             similar_labels.append(float(0))

#     return tf.data.Dataset.from_tensor_slices(similar_labels).batch(batch_size=batch_size) 


# def sliding_window(x, window_size, stride, axis=0):
#     n_in = tf.shape(x)[axis]
#     n_out = (n_in - window_size) // stride + 1
#     # Just in case n_in < window_size
#     n_out = tf.math.maximum(n_out, 0)
#     r = tf.expand_dims(tf.range(n_out), 1)
#     idx = r * stride + tf.range(window_size)
#     return tf.gather(x, idx, axis=axis)

# # images = list(train_ds.map(lambda x, y: x))
# # labels = list(train_ds.map(lambda x, y: y))

# #https://stackoverflow.com/questions/67151256/fit-a-keras-model-with-mixed-input-of-type-batchdataset-and-numpy-array
# #https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
# https://www.tensorflow.org/tutorials/load_data/images
# https://stackoverflow.com/questions/62558696/how-do-i-re-batch-a-tensor-in-tensorflow
# https://stackoverflow.com/questions/61642569/batchdataset-not-subscriptable-when-trying-to-format-python-dictionary-as-table
# https://stackoverflow.com/questions/61959517/how-to-train-keras-model-with-multiple-inputs-in-tensorflow-2-2

#     # model #

# epoch = 100
# image_size = (250,250)
# batch_size = 32
# margin = 1

#     # split data #

# half50 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half",
#     seed=1000,
#     color_mode='grayscale',
#     image_size=image_size, # size of 1 image corresponds with 250 x 250 segments 
#     batch_size=batch_size,
#     shuffle=True,
# )

# half100 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half",
#     seed=1500,
#     color_mode='grayscale',
#     image_size=image_size, 
#     batch_size=batch_size,
#     shuffle=True,
# )

# def get_label():

#     half50_labels = []
#     half100_labels = [] 
#     similar_labels = []

#     for batch in half50:
#         for label in batch[1]:
#             half50_labels.append(label.numpy())

#     for batch in half100:
#         for label in batch[1]:
#             half100_labels.append(label.numpy())

#     for index, label in enumerate(half50_labels):
#         if label == half100_labels[index]:
#             similar_labels.append(1)
#         else:
#             similar_labels.append(0)

#     similar_labels = tf.convert_to_tensor(similar_labels)
    
#     return np.asarray(similar_labels)

# image_array0 = []

# for index, element in enumerate(half50.as_numpy_iterator()):
#     image_array0.append(element[0])

# image_array0 = np.array([np.array(val) for val in image_array0],dtype=object)

# image_array1 = []

# for index, element in enumerate(half100.as_numpy_iterator()):
#     image_array1.append(element[0])

# image_array1 = np.array([np.array(val) for val in image_array1],dtype=object)

# image_list = [image_array0,image_array1]

# # print(train_dataset, type(train_dataset), test_dataset, type(test_dataset))

# similar_labels = get_label()

# # we put [half50,half100] into input, labels will be similar_labels list

# def euclidean_distance(vects):
#     """Find the Euclidean distance between two vectors.

#     Arguments:
#         vects: List containing two tensors of same length.

#     Returns:
#         Tensor containing euclidean distance
#         (as floating point value) between vectors.
#     """

#     x, y = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# input = tf.keras.layers.Input((250, 250, 1))
# x = tf.keras.layers.BatchNormalization()(input)
# x = tf.keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = tf.keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = tf.keras.layers.Flatten()(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(10, activation="tanh")(x)
# embedding_network = keras.Model(input, x)


# input_1 = tf.keras.layers.Input((250, 250, 1))
# input_2 = tf.keras.layers.Input((250, 250, 1))

# # As mentioned above, Siamese Network share weights between
# # tower networks (sister networks). To allow this, we will use
# # same embedding network for both tower networks.
# tower_1 = embedding_network(input_1)
# tower_2 = embedding_network(input_2)

# merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
# siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# def loss(margin=1):
#     """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

#   Arguments:
#       margin: Integer, defines the baseline for distance for which pairs
#               should be classified as dissimilar. - (default is 1).

#   Returns:
#       'constrastive_loss' function with data ('margin') attached.
#   """

#     # Contrastive loss = mean( (1-true_value) * square(prediction) +
#     #                         true_value * square( max(margin-prediction, 0) ))
#     def contrastive_loss(y_true, y_pred):
#         """Calculates the constrastive loss.

#       Arguments:
#           y_true: List of labels, each label is of type float32.
#           y_pred: List of predictions of same length as of y_true,
#                   each label is of type float32.

#       Returns:
#           A tensor containing constrastive loss as floating point value.
#       """

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#         return tf.math.reduce_mean(
#             (1 - y_true) * square_pred + (y_true) * margin_square
#         )

#     return contrastive_loss

# siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
# siamese.summary()

# history = siamese.fit(
#     image_list,
#     similar_labels,
#     batch_size=batch_size,
#     epochs=epoch,
# )

# siamese.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model")

# #-----------------------------------------------------------------------------------------------------------------------------#

# epoch = 50
# image_size = (250,250)
# batch_size = 32
# margin = 1

#     # split data #

# half50 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half",
#     seed=1000,
#     color_mode='grayscale',
#     image_size=image_size, # size of 1 image corresponds with 250 x 250 segments 
#     batch_size=batch_size,
#     shuffle=True,
# )

# half100 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half",
#     seed=1500,
#     color_mode='grayscale',
#     image_size=image_size, 
#     batch_size=batch_size,
#     shuffle=True,
# )

# half50_labels = []
# half100_labels = [] 
# similar_labels_list = []

# for batch in half50:
#     for label in batch[1]:
#         half50_labels.append(label.numpy())

# for batch in half100:
#     for label in batch[1]:
#         half100_labels.append(label.numpy())

# for index, label in enumerate(half50_labels):
#     if label == half100_labels[index]:
#         similar_labels_list.append(float(1))     
#     else:
#         similar_labels_list.append(float(0))

# similar_label = tf.data.Dataset.from_tensor_slices(similar_labels_list).batch(batch_size=batch_size)

#     # image generator #

# class JoinedGen(tf.keras.utils.Sequence):
#     def __init__(self, input_gen1, input_gen2, target_gen):
#         self.gen1 = input_gen1
#         self.gen2 = input_gen2
#         self.gen3 = target_gen
#         self.size = int(tf.data.experimental.cardinality(self.gen1))

#         assert len(input_gen1) == len(input_gen2) == len(target_gen)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, i):
#         i = rd.randint(0,self.size)
#         for images, labels in self.gen1.take(i):
#             self.x1 = images
#         for images, labels in self.gen2.take(i):
#             self.x2 = images
#         for labels in self.gen3.take(i):
#             self.y = labels

#         return [self.x1, self.x2], self.y

# my_gen = JoinedGen(half50, half100, similar_label)

# def euclidean_distance(vects):

#     x, y = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# def loss(margin=1):

#     def contrastive_loss(y_true, y_pred):

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#         return tf.math.reduce_mean(
#             ((1 - y_true) * square_pred) + ((y_true) * margin_square)
#         )

#     return contrastive_loss

#     # first model -> accepts inputs and embeds them#

# embed_input = tf.keras.layers.Input((250,250,1))

# x = tf.keras.layers.BatchNormalization()(embed_input)

# x = tf.keras.layers.Conv2D(16,(10,10),strides=(3,3),activation='tanh')(x)

# x = tf.keras.layers.Conv2D(32,(3,3),strides=(3,3),activation='tanh')(x)
# x = tf.keras.layers.Conv2D(64,(3,3),strides=(3,3),activation='tanh')(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=(3,3))(x)

# x = tf.keras.layers.Conv2D(128,(3,3),strides=(3,3),activation='tanh')(x)

# x = tf.keras.layers.Flatten()(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(2,activation='tanh')(x)
# embedding_network = tf.keras.Model(inputs=embed_input, outputs=x)

#     # second model -> take embedded inputs and calculates euclidean distance#

# input_1 = tf.keras.layers.Input((250,250,1))
# input_2 = tf.keras.layers.Input((250,250,1))

# # As mentioned above, Siamese Network share weights between
# # tower networks (sister networks). To allow this, we will use
# # same embedding network for both tower networks.

# tower_1 = embedding_network(input_1)
# tower_2 = embedding_network(input_2)

# merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(normal_layer)
# siamese_model = tf.keras.Model(inputs=[input_1,input_2], outputs=output_layer)

# siamese_model.compile(
#     loss=loss(margin=margin),
#     optimizer=tf.keras.optimizers.RMSprop(0.001),
#     metrics=["accuracy"]
# )

# siamese_model.summary()

# siamese_model.fit(
#     my_gen,
#     epochs=epoch,
#     verbose=1,
# )

# siamese.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model")

#-----------------------------------------------------------------------------------------------------------------------------#

# epoch = 50
# image_size = (250,250)
# batch_size = 32
# margin = 1

#     # split data #

# half50 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\First Half",
#     seed=1000,
#     color_mode='grayscale',
#     image_size=image_size, # size of 1 image corresponds with 250 x 250 segments 
#     batch_size=batch_size,
#     shuffle=True,
# )

# half100 = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\Second Half",
#     seed=1500,
#     color_mode='grayscale',
#     image_size=image_size, 
#     batch_size=batch_size,
#     shuffle=True,
# )

# half50 = half50.prefetch(buffer_size=32)
# half100 = half100.prefetch(buffer_size=32)

# half50_labels = []
# half100_labels = [] 
# similar_labels_list = []

# for batch in half50:
#     for label in batch[1]:
#         half50_labels.append(label.numpy())

# for batch in half100:
#     for label in batch[1]:
#         half100_labels.append(label.numpy())

# for index, label in enumerate(half50_labels):
#     if label == half100_labels[index]:
#         similar_labels_list.append(float(1))     
#     else:
#         similar_labels_list.append(float(0))

# similar_label = tf.data.Dataset.from_tensor_slices(similar_labels_list).batch(batch_size=batch_size)

#     # image generator #

# class Joiner(tf.keras.utils.Sequence):
#     def __init__(self, input_gen1, input_gen2, target_gen):
#         self.gen1 = input_gen1
#         self.gen2 = input_gen2
#         self.gen3 = target_gen
#         self.size = int(tf.data.experimental.cardinality(self.gen1))

#         assert len(input_gen1) == len(input_gen2) == len(target_gen)

#     def __len__(self):
#         return self.size

#     def __getitem__(self):
#         i = rd.randint(0,self.size)
#         for images, labels in self.gen1.take(i):
#             self.x1 = images
#         for images, labels in self.gen2.take(i):
#             self.x2 = images
#         for labels in self.gen3.take(i):
#             self.y = labels

#         return [self.x1, self.x2], self.y

# joined = Joiner(half50,half100,similar_label)

# def euclidean_distance(vects):

#     vect1, vect2 = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(vect1 - vect2), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# def loss(margin=1):

#     def contrastive_loss(y_true, y_pred):

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#         return tf.math.reduce_mean(
#             (y_true * square_pred) + ((1 - y_true) * (margin_square))
#         )

#     return contrastive_loss

#     # first model -> accepts inputs and embeds them#

# input_embedding = tf.keras.layers.Input((250,250,1),name='embedding')

# x = tf.keras.layers.BatchNormalization()(input_embedding)

# x = tf.keras.layers.Conv2D(32,(10,10),activation='tanh')(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(x)

# x = tf.keras.layers.Conv2D(64,(7,7),activation='tanh')(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(2,activation='tanh')(x)

# embedding_network = tf.keras.Model(inputs=input_embedding,outputs=x)

#     # second model -> take embedded inputs and calculates euclidean distance#

# input_siamese_1 = tf.keras.layers.Input((250,250,1)) 
# input_siamese_2 = tf.keras.layers.Input((250,250,1))

# # As mentioned above, Siamese Network share weights between
# # tower networks (sister networks). To allow this, we will use
# # same embedding network for both tower networks.

# tower_1 = embedding_network(input_siamese_1)
# tower_2 = embedding_network(input_siamese_2)

# merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(normal_layer)

# siamese_model = tf.keras.Model(inputs=[input_siamese_1,input_siamese_2],outputs=output_layer,name='siamese')

# embedding_network.summary()
# siamese_model.summary()

# siamese_model.compile(
#     loss=loss(margin=margin),
#     optimizer='RMSprop',
#     metrics=["accuracy"]
# )

# siamese_model.fit(
#     joined,
#     epochs=epoch,
#     verbose=1,
# )

# siamese_model.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model")

    # SECTION our model #

# # https://github.com/keras-team/keras/issues/9595

import random as rd
import tensorflow as tf
import numpy as np
import os

    # load image #

img_list = [] #a list with all images and labels as (img,label)

from os import listdir
from os.path import isfile, join
from PIL import Image

input_path = "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese"
folder_name = [f for f in listdir(input_path)]
for folder_index, folder in enumerate(folder_name):
    condition_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\{folder}"
    file_names = [f for f in listdir(condition_path) if isfile(join(condition_path, f))]
    for file in file_names:
        file_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input Siamese\\{folder}\\{file}"
        img = Image.open(file_path).convert('L') #load as grayscale
        img = np.asarray(img)
        img = img[::2,::2] #downsampling by 2
        img_tuple = (img,folder_index) #img array and label
        img_list.append(img_tuple)

    # visualize #

# from matplotlib.image import imread
# import matplotlib.pyplot as plt

# plt.imshow(img_list[0][0])
# plt.show()

    # pair #

def pair(img_list):

    images = []
    labels = []

    for image, label in img_list:
        images.append(image)
        labels.append(label)
    
    pairs = []
    pairs_labels = []

    for image_index in range(len(images)):

        image1 = images[image_index]
        label1 = labels[image_index]

        searching = True
        while searching:
            random = rd.randint(0,len(img_list)-1)

            image2 = images[random]
            label2 = labels[random]

            if image_index == 0:
                if label1 == label2:
                    pairs_labels.append(1)
                    pairs.append([image1,image2])
                    searching = False
                elif label1 != label2:
                    pairs_labels.append(0)
                    pairs.append([image1,image2])
                    searching = False                

            if (label1 == label2) and (pairs_labels[-1] != 1):
                pairs_labels.append(1)
                pairs.append([image1,image2])
                searching = False
            elif (label1 != label2) and (pairs_labels[-1] != 0):
                pairs_labels.append(0)
                pairs.append([image1,image2])
                searching = False

    pair_array = np.array(pairs)
    label_array = np.array(pairs_labels).astype("float32")

    indices = np.arange(pair_array.shape[0])
    np.random.shuffle(indices)

    pair_array = pair_array[indices]
    label_array = label_array[indices]

    return pair_array, label_array

pairs, labels = pair(img_list)

    # slice data #

x1 = pairs[:,0] 
x2 = pairs[:,1]

#     # visualize #

# import matplotlib.pyplot as plt

# def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
#     """Creates a plot of pairs and labels, and prediction if it's test dataset.

#     Arguments:
#         pairs: Numpy Array, of pairs to visualize, having shape
#                (Number of pairs, 2, 28, 28).
#         to_show: Int, number of examples to visualize (default is 6)
#                 `to_show` must be an integral multiple of `num_col`.
#                  Otherwise it will be trimmed if it is greater than num_col,
#                  and incremented if if it is less then num_col.
#         num_col: Int, number of images in one row - (default is 3)
#                  For test and train respectively, it should not exceed 3 and 7.
#         predictions: Numpy Array of predictions with shape (to_show, 1) -
#                      (default is None)
#                      Must be passed when test=True.
#         test: Boolean telling whether the dataset being visualized is
#               train dataset or test dataset - (default False).

#     Returns:
#         None.
#     """

#     # Define num_row
#     # If to_show % num_col != 0
#     #    trim to_show,
#     #       to trim to_show limit num_row to the point where
#     #       to_show % num_col == 0
#     #
#     # If to_show//num_col == 0
#     #    then it means num_col is greater then to_show
#     #    increment to_show
#     #       to increment to_show set num_row to 1
#     num_row = to_show // num_col if to_show // num_col != 0 else 1

#     # `to_show` must be an integral multiple of `num_col`
#     #  we found num_row and we have num_col
#     #  to increment or decrement to_show
#     #  to make it integral multiple of `num_col`
#     #  simply set it equal to num_row * num_col
#     to_show = num_row * num_col

#     # Plot the images
#     fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
#     for i in range(to_show):

#         # If the number of rows is 1, the axes array is one-dimensional
#         if num_row == 1:
#             ax = axes[i % num_col]
#         else:
#             ax = axes[i // num_col, i % num_col]

#         ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
#         ax.set_axis_off()
#         if test:
#             ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
#         else:
#             ax.set_title("Label: {}".format(labels[i]))
#     if test:
#         plt.tight_layout(rect=(0, 0, 1, 1), w_pad=0.0)
#     else:
#         plt.tight_layout(rect=(0, 0, 1, 1))

#     plt.show()

# visualize(pairs[:-1], labels[:-1], to_show=4, num_col=8)

    # model #

def euclidean_distance(vects):

    vect1, vect2 = vects
    sum_square = tf.math.reduce_sum(tf.math.square(vect1 - vect2), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):

    def contrastive_loss(y_true, y_pred):

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (y_true * margin_square) + ((1 - y_true) * (square_pred))
        )

    return contrastive_loss

    # first model -> accepts inputs and embeds them #

input_embedding = tf.keras.layers.Input((250,250,1),name='embedding')

x = tf.keras.layers.BatchNormalization()(input_embedding)

x = tf.keras.layers.Conv2D(16,(1,1),activation='tanh')(x)

x = tf.keras.layers.Conv2D(32,(3,3),activation='tanh')(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(3,3))(x)

x = tf.keras.layers.Conv2D(64,(1,1),activation='tanh')(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(5,5))(x)

x = tf.keras.layers.Conv2D(128,(3,3),activation='tanh')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(3,activation='tanh')(x)

embedding_network = tf.keras.Model(inputs=input_embedding,outputs=x,name='embedding')

    # second model -> take embedded inputs and calculates euclidean distance #

input_siamese_1 = tf.keras.layers.Input((250,250,1)) 
input_siamese_2 = tf.keras.layers.Input((250,250,1))

tower_1 = embedding_network(input_siamese_1)
tower_2 = embedding_network(input_siamese_2)

merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(normal_layer)

siamese_model = tf.keras.Model(inputs=[input_siamese_1,input_siamese_2],outputs=output_layer,name='siamese')

embedding_network.summary()
siamese_model.summary()

epoch = 250
margin = 1

siamese_model.compile(
    loss=loss(margin=margin),
    optimizer=tf.keras.optimizers.Adadelta(0.002),
    metrics=['accuracy']
)

class CustomCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        if logs.get('accuracy') >= 95e-2:
            self.model.stop_training = True

callback = CustomCallBack()

siamese_model.fit(
    [x1,x2],
    labels,
    epochs=epoch,
    verbose=1,
    callbacks=[callback]
)

# siamese_model.save_weights("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model Weight\\Siamese_Weights.h5")
# siamese_model.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model")

#-----------------------------------------------------------------------------------------------------------------------------#

# import random as rd
# import tensorflow as tf
# import numpy as np
# import os

# from os import listdir
# from os.path import isfile, join
# from PIL import Image

#     # SECTION load image #

# machine_prediction = True
# condition_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Condition"
# condition_names = [f.replace('.png','') for f in listdir(condition_path) if isfile(join(condition_path, f))]

# def img_loader(machine=None):

#     if machine_prediction == False:

#         img_list = [] #a list with all images and labels as (img,label)

#         # load test graph #

#         input_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Siamese"
#         folder_name = [f for f in listdir(input_path)]
#         for folder_index, folder in enumerate(folder_name):
#             condition_path = f"{input_path}\\{folder}"
#             file_names = [f for f in listdir(condition_path) if isfile(join(condition_path, f))]
#             for file in file_names:
#                 file_path = f"{input_path}\\{folder}\\{file}"
#                 img = Image.open(file_path).convert('L') #load as grayscale
#                 img = np.asarray(img)
#                 img = img[::2,::2] #downsampling by 2
#                 img_tuple = (img,folder_index) #img array and label
#                 img_list.append(img_tuple)

#         return img_list, None

#     if machine_prediction == True:

#         condition_img_list = []

#         machine_img_list = []

#         machine_list = [
#             'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
#             'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
#             'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
#             'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD',
#         ]

#         if machine == None:
#             machine = rd.choice(machine_list)

#         elif machine != None:
#             machine = machine

#         # load machine graph #

#         input_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images"

#         machine_path = f"{input_path}\\Machine"
#         machine_img_path = f"{machine_path}\\{machine}.png"
#         machine_img = Image.open(machine_img_path).convert('L') #load as grayscale
#         machine_img = np.asarray(machine_img)
#         machine_img = machine_img[::2,::2] #downsampling by 2

#         machine_img_list.append(machine_img)

#         condition_path = f"{input_path}\\Condition"
#         condition_names = [f for f in listdir(condition_path) if isfile(join(condition_path, f))]
#         for condition in condition_names:
#             file_path = f"{condition_path}\\{condition}"
#             condition_img = Image.open(file_path).convert('L') #load as grayscale
#             condition_img = np.asarray(condition_img)
#             condition_img = condition_img[::2,::2] #downsampling by 2
#             condition_img_list.append(condition_img)            

#         return machine_img_list, condition_img_list

# machine_img_list, condition_img_list = img_loader(machine='E3CE')

#     # SECTION pair #

# def pair(machine_img_list,condition_img_list):

#     if machine_prediction == False:

#         images = []
#         labels = []

#         for image, label in machine_img_list:
#             images.append(image)
#             labels.append(label)
        
#         pairs = []
#         pairs_labels = []

#         for image_index in range(len(images)):

#             image1 = images[image_index]
#             label1 = labels[image_index]

#             searching = True
#             while searching:
#                 random = rd.randint(0,len(machine_img_list)-1)

#                 image2 = images[random]
#                 label2 = labels[random]

#                 if image_index == 0:
#                     if label1 == label2:
#                         pairs_labels.append(1)
#                         pairs.append([image1,image2])
#                         searching = False
#                     elif label1 != label2:
#                         pairs_labels.append(0)
#                         pairs.append([image1,image2])
#                         searching = False                

#                 if (label1 == label2) and (pairs_labels[-1] != 1):
#                     pairs_labels.append(1)
#                     pairs.append([image1,image2])
#                     searching = False
#                 elif (label1 != label2) and (pairs_labels[-1] != 0):
#                     pairs_labels.append(0)
#                     pairs.append([image1,image2])
#                     searching = False

#         pair_array = np.array(pairs)
#         label_array = np.array(pairs_labels).astype("float32")

#         indices = np.arange(pair_array.shape[0])
#         np.random.shuffle(indices)

#         pair_array = pair_array[indices]
#         label_array = label_array[indices]

#         return pair_array, label_array

#     if machine_prediction == True:
        
#         pairs = []

#         for condition_image in condition_img_list:

#             pairs.append([machine_img_list[0],condition_image])

#         pair_array = np.array(pairs)

#         return pair_array, None

# pairs_test, labels_test = pair(machine_img_list,condition_img_list)

#     # SECTION shuffle data #

# x1_test = pairs_test[:,0] 
# x2_test = pairs_test[:,1]

#     # SECTION model #

# def euclidean_distance(vects):

#     vect1, vect2 = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(vect1 - vect2), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# def loss(margin=1):

#     def contrastive_loss(y_true, y_pred):

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#         return tf.math.reduce_mean(
#             (y_true * margin_square) + ((1 - y_true) * (square_pred))
#         )

#     return contrastive_loss

#     # first model -> accepts inputs and embeds them #

# input_embedding = tf.keras.layers.Input((250,250,1),name='embedding')

# x = tf.keras.layers.BatchNormalization()(input_embedding)

# x = tf.keras.layers.Conv2D(16,(1,1),activation='tanh')(x)

# x = tf.keras.layers.Conv2D(32,(3,3),activation='tanh')(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=(3,3))(x)

# x = tf.keras.layers.Conv2D(64,(1,1),activation='tanh')(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=(5,5))(x)

# x = tf.keras.layers.Conv2D(128,(3,3),activation='tanh')(x)
# x = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# x = tf.keras.layers.Dense(3,activation='tanh')(x)

# embedding_network = tf.keras.Model(inputs=input_embedding,outputs=x,name='embedding')

#     # second model -> take embedded inputs and calculates euclidean distance #

# input_siamese_1 = tf.keras.layers.Input((250,250,1)) 
# input_siamese_2 = tf.keras.layers.Input((250,250,1))

# tower_1 = embedding_network(input_siamese_1)
# tower_2 = embedding_network(input_siamese_2)

# merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(normal_layer)

# siamese_model = tf.keras.Model(inputs=[input_siamese_1,input_siamese_2],outputs=output_layer,name='siamese')

# embedding_network.summary()
# siamese_model.summary()

# epoch = 250
# margin = 1
# batch_size=32

# siamese_model.compile(
#     loss=loss(margin=margin),
#     optimizer=tf.keras.optimizers.Adadelta(0.002),
#     metrics=['accuracy']
# )

# siamese_model.load_weights("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Similarity Model Weight\\Siamese_Weights.h5")

#     # SECTION visualize #

# import matplotlib.pyplot as plt

# def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):

#     num_row = to_show // num_col if to_show // num_col != 0 else 1

#     to_show = num_row * num_col

#     fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
#     for index in range(to_show):

#         if num_row == 1:
#             ax = axes[index % num_col]
#         else:
#             ax = axes[index // num_col, index % num_col]

#         ax.imshow(tf.concat([pairs[index][0], pairs[index][1]], axis=1), cmap="gray")
#         ax.set_axis_off()

#         if machine_prediction == False:
#             if test:
#                 ax.set_title("True: {} | Pred: {:.5f}".format(labels[index],predictions[index][0]))
#             else:
#                 ax.set_title("Label: {}".format(labels[index]))
#         else:
#             ax.set_title("Pred: {:.5f} | {}".format(predictions[index][0],condition_names[index]))            
            
#     if test:
#         plt.tight_layout(rect=(0, 0, 1, 1), w_pad=0.0)
#     else:
#         plt.tight_layout(rect=(0, 0, 1, 1))

#     plt.show()

# predictions = siamese_model.predict([x1_test, x2_test])

# if machine_prediction == False:
#     to_show = 9
# else:
#     to_show = 3

# visualize(pairs_test, labels_test, to_show=to_show, predictions=predictions, test=True)

    # SECTION graph feature map #

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

# layer_outputs = [layer.output for layer in embedding_network.layers[1:]]

# visualize_model = tf.keras.models.Model(inputs = embedding_network.input, outputs = layer_outputs)

# machine_path = r"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Machine\\E3CE.png"
# condition_path = r"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Condition\\Loose.png"

# machine_img = tf.keras.preprocessing.image.load_img(
#     machine_path,
#     color_mode='grayscale',
#     target_size=(250,250),    
# )

# condition_img = tf.keras.preprocessing.image.load_img(
#     condition_path,
#     color_mode='grayscale',
#     target_size=(250,250),    
# )

# img_list = [machine_img,condition_img]
# name_list = ['E3CE','Loose']

# for index, img_set in enumerate(img_list):

#     img = tf.keras.preprocessing.image.img_to_array(img_set)

#     img = img.reshape((1,250,250,1))

#     img = img / 255

#     feature_maps = visualize_model.predict(img)

#     layer_names = [layer.name for layer in embedding_network.layers]

#     import numpy as np
#     import matplotlib.pyplot as plt

#     count = 0

#     for layer_names, feature_maps in zip(layer_names,feature_maps):

#         count += 1

#         if len(feature_maps.shape) == 4:
#             channels = feature_maps.shape[-1] # 1 channel for 1 filter
#             size = feature_maps.shape[1] 
#             display_grid = np.zeros((size, size * channels))
#             print(size, channels)

#             for i in range(channels):
#                 x = feature_maps[0,:,:,i]
#                 x -= x.mean()           
#                 x *= 125
#                 x += 250
#                 x = np.clip(x,0,255).astype('uint8')
#                 display_grid[:,i*size:(i+1)*size] = x
            
#             scale = 25. / channels

#             fig = plt.figure(figsize=(scale*channels,scale),dpi=150)
#             plt.grid(False)
#             plt.axis('off')
#             plt.imshow(display_grid,aspect='auto',cmap='viridis')

#             fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Generated\\Visualize Features\\{name_list[index]}-{count}-{layer_names}.png")
#             fig.clf()

    # SECTION visualize 2.0 #

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout(rect=(0, 0, 0, 0))
# fig.canvas.manager.full_screen_toggle()
# plt.subplots_adjust(wspace=0.0,hspace=0.0)

# # load the model
# model = embedding_network   

# layer_outputs = [layer.output for layer in embedding_network.layers[1:]]

# path_list = [
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Machine\\E3CE.png",
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Condition\\Loose.png",
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Condition\\Misalign.png",
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Condition\\Unbalance.png",
# ]

# name_list = [
#     'E3CE',
#     'Loose',
#     'Misalign',
#     'Unbalance'
# ]

# for index in range(len(layer_outputs)-2):

#     print(index)

#     model = embedding_network   

#     layer_outputs = [layer.output for layer in embedding_network.layers[1:]]

#     # redefine model to output right after the first hidden layer
#     model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[index].output) #output is index of cnv layer (cnv2d_1 index = 1)

#     for number, path in enumerate(path_list):

#         img = tf.keras.preprocessing.image.load_img(
#             path,
#             color_mode='grayscale',
#             target_size=(250,250),    
#         )

#         img = tf.keras.preprocessing.image.img_to_array(img)

#         img = np.expand_dims(img, axis=0)

#         img = img / 255

#         # get feature map for first hidden layer
#         feature_maps = model.predict(img)

#         # plot all 64 maps in an 8x8 squares
#         channels = feature_maps.shape[-1]
#         square = int(np.round(np.sqrt(channels)))
#         image_index = 0

#         for _ in range(square):
#             for _ in range(square):

#                 if image_index >= channels:
#                     pass
#                 else:
#                     # specify subplot and turn of axis
#                     ax = plt.subplot(square, square, image_index+1)
#                     ax.set_xticks([])
#                     ax.set_yticks([])

#                     # plot filter channel in grayscale
#                     plt.imshow(feature_maps[0, :, :, image_index], cmap='gray')    
            
#                 image_index += 1

#         fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Generated\\Visualize Features\\Pattern-{name_list[number]}-{index}.png")
#         fig.clf()

