import tensorflow as tf
from tensorflow import keras
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------------------------------------------------------#

#     # model #

# image_size = (500,500)
# batch_size = 10

#     # split data #

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
#     validation_split=0.25,
#     subset="training",
#     seed=1337,
#     color_mode='rgb',
#     image_size=image_size, # size of 1 image corresponds with 500 x 500 segments 
#     batch_size=batch_size,
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
#     validation_split=0.25,
#     subset="validation",
#     seed=1337,
#     color_mode='rgb',
#     image_size=image_size, 
#     batch_size=batch_size,
# )

# #     # visualize #

# # import matplotlib.pyplot as plt

# # plt.figure(figsize=(12, 12))
# # for images, labels in train_ds.take(1):
# #     for i in range(9):
# #         ax = plt.subplot(3, 3, i + 1)
# #         plt.imshow(images[i].numpy().astype("uint8"))
# #         plt.title(int(labels[i]))
# #         plt.axis("off")

# # plt.show()

#     # configure the dataset for performance #

# train_ds = train_ds.prefetch(buffer_size=10)
# val_ds = val_ds.prefetch(buffer_size=10)

# # for data, labels in train_ds:
# #     print(data.shape)  # (32, 300, 300, 1) -> (batch, (size), scale(RGB(3) or grayscale(1)))
# #     print(data.dtype)  # <dtype: 'float32'>
# #     print(labels.shape)  # (32,)
# #     print(labels.dtype)  # <dtype: 'int32'>

#     # make model #

# model = keras.models.Sequential([

#     keras.layers.Rescaling(1.0 / 255),

#     keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", strides=(3,3), input_shape=(500,500,3)),
#     keras.layers.Activation('relu'),
#     keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),

#     keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=(3,3)),
#     keras.layers.Activation('relu'),
#     keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),

#     keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", strides=(3,3)),
#     keras.layers.Activation('relu'),
#     keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

#     keras.layers.Flatten(), 
#     keras.layers.Dense(128),
#     keras.layers.Activation('relu'),

#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(1),
#     keras.layers.Activation('sigmoid')

# ])

# epochs = 10

# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]

# model.compile(
#     optimizer=keras.optimizers.Adam(0.0005),
#     loss='binary_crossentropy',
#     metrics=['accuracy'],
# )

# model.fit(
#     train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
# )

# model.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Deep Learning")

#-----------------------------------------------------------------------------------------------------------------------------#

#     # predict #

# model = keras.models.load_model("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Deep Learning")
# # print(model.summary())

# predict_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Classification Test 1",
#     labels='inferred',
#     label_mode='int',
#     color_mode='rgb',
#     image_size=(500,500),
#     seed=1337,
#     shuffle=False,
#     batch_size=100
# )

# label = np.concatenate([label for image, label in predict_ds], axis=0).tolist()
# print(label)
# predict = np.where(model.predict(predict_ds) > 0.5, 1,0).flatten().tolist()
# print(predict)
# predict_percent = model.predict(predict_ds).flatten().tolist()
# print(predict_percent)
# evaluation = model.evaluate(predict_ds)
# print(evaluation)

# import matplotlib.pyplot as plt
# import math 

# # for i in range(len(label)):
# #     print(label[i],predict[i])

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()
# fig.suptitle(f'Hard Test (+-70%) Loss {evaluation[0]:.2f} Accuracy {evaluation[1]:.2f}', fontsize=10)

# for images, labels in predict_ds.take(1):
#     for i in range(100):
#         ax = plt.subplot(10, 10, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(f'real {int(label[i])} | pred {int(predict[i])} | perc {float(predict_percent[i]*100):.2f}', fontsize=5)
#         plt.axis("off")

# plt.show()