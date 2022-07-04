import tensorflow as tf
from tensorflow import keras
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------------------------------------------------------#

    # model #

image_size = (250,250)
batch_size = 16

    # split data #

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
    validation_split=0.2,
    subset="training",
    seed=1000,
    color_mode='grayscale',
    image_size=image_size, # size of 1 image corresponds with 250 x 250 segments 
    batch_size=batch_size,
    shuffle=True,

)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Input",
    validation_split=0.2,
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

    keras.layers.Rescaling(1.0 / 255),

    keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(250,250,1)),
    keras.layers.Activation('relu'),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3)),
    keras.layers.Activation('relu'),
    keras.layers.AveragePooling2D(pool_size=(3,3), strides=(3,3)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)),

    keras.layers.Flatten(), 
    keras.layers.Dense(128), # 128 unique features
    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5),
    keras.layers.Dense(3), # 3 outputs 
    keras.layers.Activation('softmax')

])

epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
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

model.save("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Deep Learning")

#-----------------------------------------------------------------------------------------------------------------------------#

    # predict #

# model = keras.models.load_model("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Deep Learning")
# print(model.summary())

# predict_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Images\\Test Images",
#     labels='inferred',
#     label_mode='int',
#     color_mode='grayscale',
#     image_size=(250,250),
#     shuffle=False,
#     batch_size=100
# )

# predict_percent = model.predict(predict_ds)
# percentage_list = predict_percent.tolist()

# for index_percent, machine in enumerate(percentage_list):
#     machine_list = []
#     for index_machine, percent in enumerate(machine):
#         machine_list.append((percent/sum(machine)) * 100)
#     percentage_list[index_percent] = machine_list

# print(percentage_list)

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()
# fig.suptitle(f'Machine Diagnosis', fontsize=10)

# name_list = [
#     'E3CE', 'E3F7', 'E484', 'E3D1', 'E407', 'E40E', 'E3FC', 'E3DF', 'E3DC', 'E410',
#     'E3EC', 'E3E1', 'E3D5', 'E3E6', 'E413', 'E470', 'E401', 'E47E', 'E3D4', 'E466',
#     'E412', 'E408', 'E402', 'E3F8', 'E480', 'E3F0', 'E473', 'E3D8', 'E417', 'E3FA',
#     'E416', 'E3F3', 'E476', 'E47B', 'E3F1', 'E414', 'E3FB', 'E3FD'
# ]

# for labels in percentage_list:
#     for i in range(40):
#         ax = plt.subplot(8, 5, i+1)
#         plt.imshow(predict_ds[i].numpy().astype("uint8"))
#         # plt.title(f'Loose {float(label[i]):.2f} | Misalign {float(predict[i]):.2f} | Unbalance {float(predict_percent[i]*100):.2f}', fontsize=8)
#         plt.axis("off")

# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#

#     # predict #
# # https://www.youtube.com/watch?v=i3qjgJgQqgg

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.tight_layout()
# fig.canvas.manager.full_screen_toggle()

# model = keras.models.load_model("E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Classification Model")

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

#         fig = plt.figure(figsize=(scale*channels,scale),dpi=300)
#         plt.grid(False)
#         plt.axis('off')
#         plt.imshow(display_grid,aspect='auto',cmap='viridis')

#         fig.savefig(f"E:\\Gun's stuff\\Machine Learning and Deep Learning\\Machine Learning\\Ajinomoto Project\\Generated Images\\Visualize Features\\({count}){layer_names}.png")
#         fig.clf()
