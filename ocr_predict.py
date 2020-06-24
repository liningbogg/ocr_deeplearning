import redis
import numpy as np
import time
import json
from cacheout import Cache
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model


class Predictor_Ocr:
    def __init__(self, ip, port, password):
        self.ip = ip
        self.port = port
        self.password = password
        self.redis_pool=redis.ConnectionPool(host=ip, port=port,db=0, password=password, encoding='utf-8')
        self.cache = Cache()
        self.cur = os.getcwd()

    def get_model(self, model_key):
        if self.cache.has(model_key) is True:
            return self.cache.get(model_key)
        else:
            path = "%s/%s/data/%s.h5" % (self.cur, model_key, model_key)
            model =  load_model(path)
            self.cache.set(model_key, model)
            return model

    def work(self):
        while True:
            width = 64
            height = 64
            red = redis.Redis(connection_pool=self.redis_pool)
            task_str = red.lpop("aiocr")
            if task_str is not None:
                task = json.loads(task_str)
                create_user_id = task["create_user_id"]
                image_id = task["image_id"]
                polygon_id = task["polygon_id"]
                image = task["image"]
                algorithm_set = task["algorithm"]
                image_list = [image]
                image_batch = np.array(image_list, dtype=np.float32).reshape(-1, width, height, 1)
                image_batch = image_batch/255.0
                out_list = []
                top_k = 5
                thr = 0.1
                for algorithm in algorithm_set:
                    path = "%s/%s/data/elem_list.json" % (self.cur, algorithm)
                    with open(path,'r') as f:
                        elem_list = json.load(f)
                    model = self.get_model(algorithm)
                    out = model.predict(image_batch)[0]
                    top_candidate =out.argsort()[::-1][0:top_k]
                    for item in top_candidate:
                        if out[item] >thr and elem_list[item]>-1:
                            out_list.append(elem_list[item])
                key = "%s_%s_%s_%s" % ("rs_aiocr", create_user_id, image_id, polygon_id)
                red.set(key, json.dumps(out_list))
                
            time.sleep(0.005)

if __name__=="__main__":
    ip = "192.168.1.101"
    port = 6379
    password = "1a2a3a"
    predictor = Predictor_Ocr(ip, port, password)
    predictor.work()
