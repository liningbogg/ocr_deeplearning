import os
import numpy as np
import json
import random
from PIL import Image, ImageOps
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model
import time
import redis

model = load_model("./data/withoutdot/withoutdot.h5")

with open('./data/withoutdot/elem_list.json','r') as f:
    elem_list = json.load(f)

width = 128
height = 128

image_train = image_train/255.0
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

        
        key = "%s_%s_%s_%s" % ("rs_aiocr", create_user_id, image_id, polygon_id)

    model.predict(image_train[:1])
    time.sleep(5)
