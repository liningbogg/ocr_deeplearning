import sys
sys.path.append(".") 
import numpy as np
import json
import random
from PIL import Image, ImageOps
from keras import models, layers, optimizers
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, MaxPool2D
from keras.optimizers import SGD
from resnet import ResNet
from keras.utils import plot_model


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

with open('./data/elem_list.json','r') as f:
    elem_list = json.load(f)
with open('./data/label.json','r') as f:
    label_list = json.load(f)
random.shuffle(label_list)
label_num = len(label_list)
elem_num = len(elem_list)
train_per = 0.95
width = 64
height = 64

label_index = {}
index = 0
for elem_id in elem_list:
    label_index[elem_id]=index
    index = index + 1

print(label_index)

train_num = int(train_per*label_num)
val_num = label_num - train_num
print(label_num, train_num, val_num)
train_line = label_list[:train_num]
val_line = label_list[train_num:]
image_list = []
train_target = np.zeros((train_num, elem_num), dtype='float32')
val_target = np.zeros((val_num, elem_num), dtype='float32')
print(train_target.shape)
print(val_target.shape)
index = 0
for line in train_line:
    filename = line['file']
    image = Image.open(filename)
    image_flat = list(image.getdata())
    image_list.append(image_flat)
    elem_label = line['label']
    for elem in elem_label:
        train_target[index][label_index[elem]]=1.0
    index = index + 1

image_train = np.array(image_list, dtype=np.float32).reshape(-1, width, height, 1)

image_train = image_train/255.0

image_list_val = []
index = 0
for line in val_line:
    filename = line['file']
    image = Image.open(filename)
    image_flat = list(image.getdata())

    image_list_val.append(image_flat)
    elem_label = line['label']
    for elem in elem_label:
        val_target[index][label_index[elem]]=1.0

    index = index + 1
image_val = np.array(image_list_val, dtype=np.float32).reshape(-1, width, height,1)
image_val = image_val/255.0


model = ResNet.build(width, height, 1, elem_num, stages=[3,4,6],filters=[64,128,256,512])#因为googleNet默认输入32*32的图片
plot_model(model, to_file="data/resnet.png", show_shapes=True)

sgd = SGD(lr=0.001, decay=1e-5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(image_train, train_target, batch_size=64, epochs=2000, validation_split=0.2)
model.evaluate(image_val, val_target)
out_set = model.predict(image_val)
model.save("data/resnet.h5")
f = open("out.txt", "w") 
index = 0
for out in out_set:
    for out_elem in out:
        print("%0.2f" % out_elem, end='', file=f)
    print('\n', file=f)
    for target_elem in val_target[index]:
        print("%0.2f " % target_elem, end='', file=f)
    print('\n', file=f)
    print('-----------------', file=f)
    index = index +1

