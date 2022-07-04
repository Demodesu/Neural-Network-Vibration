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
