import tensorflow as tf
from tensorflow import keras
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------------------------------------------------------#

    # model #

image_size = (250,250)
batch_size = 32

    # split data #

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
    validation_split=0.25,
    subset="training",
    seed=1000,
    color_mode='grayscale',
    image_size=image_size, # size of 1 image corresponds with 250 x 250 segments 
    batch_size=batch_size,
    shuffle=True,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
    validation_split=0.25,
    subset="validation",
    seed=1000,
    color_mode='grayscale',
    image_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
)

#     # visualize #

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 12))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")

# plt.show()

    # configure the dataset for performance #

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

    # make model #

model = keras.models.Sequential([

    keras.layers.RandomTranslation((-0.3,0.3),(-0.3,0.05)),
    keras.layers.RandomZoom((-0.5,0.5)),
    keras.layers.RandomCrop(200,200),
    keras.layers.Rescaling(1.0/255),

    keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(3,3), input_shape=(250,250,1), padding='valid'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),

    keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'),
    keras.layers.Activation('relu'),

    keras.layers.Flatten(), 
    keras.layers.Dense(128), 
    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5),
    keras.layers.Dense(3), # 3 outputs 
    keras.layers.Activation('softmax')

])

epochs = 100

callbacks = [
    # keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    # keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, mode='max')
]

model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

model.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Classification Model 2.0")

#-----------------------------------------------------------------------------------------------------------------------------#

#     # graph feature map #

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

# model = keras.models.load_model("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Classification Model 2.0")

# layer_outputs = [layer.output for layer in model.layers[1:]]

# visualize_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

# image_path = r"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Test\\normalE3CE.png"

# img = tf.keras.preprocessing.image.load_img(
#     image_path,
#     color_mode='grayscale',
#     target_size=(250,250),    
# )

# x = tf.keras.preprocessing.image.img_to_array(img)

# x = x.reshape((1,250,250,1))

# x = x / 255

# feature_maps = visualize_model.predict(x)

# layer_names = [layer.name for layer in model.layers]

# import numpy as np
# import matplotlib.pyplot as plt

# count = 0

# for layer_names, feature_maps in zip(layer_names,feature_maps):
#     print(feature_maps.shape)
#     count += 1
#     if len(feature_maps.shape) == 4:
#         channels = feature_maps.shape[-1] # 1 channel for 1 filter
#         size = feature_maps.shape[1] 
#         display_grid = np.zeros((size, size * channels))
#         for i in range(channels):
#             x = feature_maps[0,:,:,i]
#             x -= x.mean()
#             x *= 125
#             x += 250
#             x = np.clip(x,0,255).astype('uint8')
#             display_grid[:,i*size:(i+1)*size] = x
        
#         scale = 50. / channels

#         fig = plt.figure(figsize=(scale*channels,scale),dpi=250)
#         plt.grid(False)
#         plt.axis('off')
#         plt.imshow(display_grid,aspect='auto',cmap='viridis')

#         fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Visualize Features 2.0\\({count}){layer_names}.png")
#         fig.clf()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # predict #

# model = keras.models.load_model("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Classification Model 2.0")

# predict_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images",
#     color_mode='grayscale',
#     image_size=(250,250),
#     shuffle=False,
# )

# predict_percent = model.predict(predict_ds)
# percentage_list = predict_percent.tolist()

# for index_percent, machine in enumerate(percentage_list):
#     machine_list = []
#     for index_machine, percent in enumerate(machine):
#         machine_list.append((percent/sum(machine)) * 100)
#     percentage_list[index_percent] = machine_list

# import matplotlib.pyplot as plt

# fig = plt.figure()

# name_list = [
#     'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
#     'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
#     'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
#     'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD'
# ]

# condition_list = [
#     'Loose',
#     'Misalign',
#     'Unbalance'
# ]

# cmap_list = [
#     'Blues',
#     'Reds',
#     'Greens'
# ]

# fig = plt.figure()
# fig.tight_layout()
# fig.suptitle('Blue = Loose, Red = Misalign, Green = Unbalance', fontsize=16)

# for machine_index, name in enumerate(name_list):

#     image_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images\\Test\\normal{name}.png"

#     img = tf.keras.preprocessing.image.load_img(
#         image_path,
#         color_mode='grayscale',
#         target_size=(250,250),    
#     )

#     x = tf.keras.preprocessing.image.img_to_array(img)

#     ax = plt.subplot(8, 5, machine_index+1)
#     ax.set_title(f'{name} {percentage_list[machine_index][0]:.2f} {percentage_list[machine_index][1]:.2f} {percentage_list[machine_index][2]:.2f}',fontsize=5)
#     ax.imshow(x)

#     for condition_index, image in enumerate(condition_list):

#         img_path = f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Classification\\{image}.png"

#         img = tf.keras.preprocessing.image.load_img(
#             img_path,
#             color_mode='grayscale',
#             target_size=(250,250),    
#         )

#         img = tf.keras.preprocessing.image.img_to_array(img)
#         ax.imshow(img, cmap=cmap_list[condition_index], alpha=percentage_list[machine_index][condition_index]/(sum(percentage_list[machine_index])*1.5))

#     ax.grid(False)
#     ax.axis('off')

# plt.show()