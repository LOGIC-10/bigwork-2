import csv
import os
from unittest import result
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

predict_generator = predict_datagen.flow_from_directory(
        # 种类文件夹的路径
        'predict',
        # 目标图片大小
        target_size=(20,20),
        # 目标颜色模式
        color_mode="grayscale",
        # 种类名字
        classes=None,
        # 种类模式：分类
        class_mode='categorical',
        # batch_size
        batch_size=1,
        # shuffle
        shuffle=False
)

model = tf.keras.Sequential([
    #(-1,20,20,1)->(-1,20,20,32)
    tf.keras.layers.Conv2D(input_shape=(20, 20, 1),filters=32,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,20,20,32)->(-1,10,10,32)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,10,10,32)->(-1,10,10,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,10,10,64)->(-1,5,5,64)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,5,5,64)->(-1,5,5,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,5,5,64)->(-1,5*5*64)
    tf.keras.layers.Flatten(),
    #(-1,5*5*64)->(-1,256)
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #(-1,256)->(-1,31)
    tf.keras.layers.Dense(31, activation=tf.nn.softmax)
])

print(model.summary())

model.load_weights('weights/best')
label_index = []
with open(r'weights/classes.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)#忽略第一行
    for line in reader:
        label_index.append(line[0])

file = 'results.csv'
results = []

with open(file, 'w', encoding='utf_8_sig',newline='') as csv_file:
    print('\n--- printing results to file {}'.format(file))
    writer = csv.writer(csv_file)
    writer.writerow(['picture', 'label'])
    predict_path = r'predict'
    picture_list = [f for f in os.listdir(predict_path) if not f.startswith('.')]
    picture_list.sort(key=lambda x: int(x[:-4]))   
    for picture in picture_list:
        img = cv2.imread('predict/' + picture, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (20, 20))
        img_arr = img / 255.0
        img_arr = img_arr.reshape((1, 20, 20, 1))

        y = model.predict(img_arr, batch_size=1)
        res = label_index[y.argmax()]
        results.append(res)
        writer.writerow([picture, res])

# 计算准确率
ground_truth_csv = r'ground_truth.csv'
ground_truth_dic = {}
with open(ground_truth_csv, 'r') as f:
    print('\n--- reading ground truths from file {}'.format(ground_truth_csv))
    reader = csv.reader(f)
    right = 0

    next(reader)#忽略第一行

    i = 0
    for line in reader:
        if(results[i] == line[1]):
            right += 1
        i += 1
    
    print('predict accuracy: ', right / len(picture_list))
