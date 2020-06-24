import os
import numpy as np
import json
import random
from PIL import Image, ImageOps
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model
import time
import redis

model_withoutdot = load_model("./data/withoutdot/withoutdot.h5")
model_dot = load_model("./data/dot/dot.h5")
with open('./data/withoutdot/elem_list.json','r') as f:
        elem_list = json.load(f)

width = 128
height = 128

redis_pool=redis.ConnectionPool(host='192.168.1.101', port=6379,db=0, password='1a2a3a', encoding='utf-8')
while True:
    red = redis.Redis(connection_pool=redis_pool)
    task_str = red.lpop("aiocr")
    if task_str is not None:
        task = json.loads(task_str)
        create_user_id = task["create_user_id"]
        image_id = task["image_id"]
        polygon_id = task["polygon_id"]
        image = task["image"] 
        image_list = [image]
        image_batch = np.array(image_list, dtype=np.float32).reshape(-1, width, height, 1)
        image_batch = image_batch/255.0
        print(image_batch.shape)
        out_dot = model_dot.predict(image_batch)
        out_withoutdot = model_withoutdot.predict(image_batch)
        out_list = []
        class_dot= np.argmax(out_dot)
        top_k = 5
        thr = 0.1
        if class_dot == 1:
            out_list.append(408)  # ,
        if class_dot == 2:
            out_list.append(407)  # 。

        top_candidate = out_withoutdot[0].argsort()[::-1][0:top_k]
        for candidate in top_candidate:
            if out_withoutdot[0][candidate] > thr:
                out_list.append(elem_list[candidate])
        
        key = "%s_%s_%s_%s" % ("rs_aiocr", create_user_id, image_id, polygon_id)
        print(key)
        red.set(key, json.dumps(out_list))

    time.sleep(0.005)
