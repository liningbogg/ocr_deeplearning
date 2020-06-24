import redis
from PIL import Image, ImageOps
from progressbar import *
import io
import math
import json
import pymysql
import pickle

'''
'''

def rotate_points(points, rotate_degree, w, h):
    for point in points:
        x = point['x']
        y = point['y']
        x_shift = x - w/2.0
        y_shift = y - h/2.0
        rotate_rad = rotate_degree/180.0*math.pi
        nx = x_shift*math.cos(rotate_rad)+y_shift*math.sin(rotate_rad)+w/2.0
        ny = -x_shift*math.sin(rotate_rad)+y_shift*math.cos(rotate_rad)+h/2.0
        point['x']=nx
        point['y']=ny


def bounding_points(points):
    x_min = 10000000
    y_min = 10000000
    x_max = -10000000
    y_max = -10000000
    for point in points:
        x = point["x"]
        y = point["y"]
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    points[0]["x"] = x_min
    points[0]["y"] = y_min
    points[1]["x"] = x_max
    points[1]["y"] = y_min
    points[2]["x"] = x_max
    points[2]["y"] = y_max
    points[3]["x"] = x_min
    points[3]["y"] = y_max


def get_rect_info(point_a, point_b):
    try:
        x = min(point_a['x'], point_b['x'])
        x_ = max(point_a['x'], point_b['x'])
        y = min(point_a['y'], point_b['y'])
        y_ = max(point_a['y'], point_b['y'])
        w = abs(x_-x)
        h = abs(y_-y)
        area = w*h
        return {'x':x, 'y':y, 'x_':x_, 'y_':y_, 'w':w, 'h':h, 'area':area}
    except Exception as e:
        return None



def cal_rotate_angle(x0, y0, x3, y3):

    """
    calulate rotate angle from new position
    x0, y0: position 0 rotated
    x3, y3: position 3 rotated
    """
    try:
        angle = -math.atan((x0-x3)*1.0/(y0-y3))*180/math.pi
        return angle
    except Exception as e:
        return 0


def data_achieve(hostid, host_user, database, host_passwd):

    '''
    '''
    
    redis_pool=redis.ConnectionPool(host='127.0.0.1', port=6379,db=0, password='1a2a3a', encoding='utf-8')
    red = redis.Redis(connection_pool=redis_pool)
    elem_list = []
    polygon_elem = {}
    with open("./data/elem_list.txt", "r") as f:
        for line in f:
            line = line.strip('\n')
            elem_info = json.loads(line)
            elem_id = elem_info["id"]
            elem_list.append(elem_id)
    connection = pymysql.connect(host=hostid, user=host_user, password=host_passwd, charset="utf8", use_unicode=True)
    db_cursor = connection.cursor()
    connection.select_db(database)

    for elem_id in elem_list:
        sql_polygon_id = 'select polygon_id from ocr_polygonelem where elem_id = (%s)'
        db_cursor.execute(sql_polygon_id, elem_id)
        polygon_id_set = db_cursor.fetchall()
        for polygon_id in polygon_id_set:
            polygonid = polygon_id[0]
            if polygonid in polygon_elem:
                polygon_elem[polygonid].append(elem_id)
            else:
                polygon_elem[polygonid] = [elem_id]

    data_label = []
    elem_set = set()
    widgets = ['data_achieving: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ']
    bar_achieve_label = progressbar.ProgressBar(widgets=widgets, maxval=len(polygon_elem))
    bar_achieve_label.start()
    process_step = 0
    for key in polygon_elem:
        # 获取polygon信息
        sql_polygon = 'select labeling_content, pdfImage_id, polygon from ocr_ocrlabelingpolygon where id = %s'
        db_cursor.execute(sql_polygon, key)
        labeling_content, pdfImage_id, polygon = db_cursor.fetchone()
        if labeling_content == 0:
            process_step = process_step+1
            continue

        points = json.loads(str(polygon, encoding = "utf8"))
        degree_to_rotate = cal_rotate_angle(points[0]['x'], points[0]['y'], points[3]['x'], points[3]['y'])
        degree_to_rotate = round(degree_to_rotate, 1)
        key_image_rotated = "%s_%s" % (pdfImage_id, degree_to_rotate)
        image_rotated = red.get(key_image_rotated)
        
        if image_rotated is None:
            # 获取原始图片image
            sql_pdfimage = 'select width, height, data_byte from ocr_pdfimage where id = %s'
            db_cursor.execute(sql_pdfimage, pdfImage_id)
            width, height, data_byte = db_cursor.fetchone()
            data_stream=io.BytesIO(data_byte)
            pil_image = Image.open(data_stream)
            gray = pil_image.convert('L')
            if abs(degree_to_rotate)<0.01:
                image_rotated = gray
            else:
                image_rotated = gray.rotate(degree_to_rotate)
            image_rotated = pickle.dumps(image_rotated)
            red.set(key_image_rotated, image_rotated)
        image_rotated = pickle.loads(image_rotated)
        pdf_width, pdf_height = image_rotated.size
        rotate_points(points, degree_to_rotate, pdf_width, pdf_height)
        bounding_points(points)
        rect_info = get_rect_info(points[0], points[2])
        rect = (rect_info['x'], rect_info['y'], rect_info['x_'], rect_info['y_'])
        image_corp = image_rotated.crop(rect)
        width ,height = image_corp.size
        const_width = 64
        const_height = 64
        ratio_w =width*1.0/const_width
        ratio_h =height*1.0/const_height
        ratio = max(ratio_w, ratio_h)
        tar_width = min(int(width/ratio),const_width)
        tar_height = min(int(height/ratio),const_height)
        image_resized = image_corp.resize((tar_width, tar_height), Image.ANTIALIAS)
        tar_width, tar_height = image_resized.size
        w_extend = const_width - tar_width
        h_extend = const_height - tar_height
        w_l_extend = w_extend // 2
        w_r_extend = w_extend -w_l_extend
        h_t_extend = h_extend // 2
        h_b_extend = h_extend -h_t_extend


        image_padding = ImageOps.expand(image_resized, border=(w_l_extend, h_t_extend, w_r_extend, h_b_extend) ,fill=0)
        image_file = "./data/image/%s.png" % key
        image_padding.save(image_file)
        label= {'file': image_file, 'label': polygon_elem[key]}
        data_label.append(label)
        for elem in polygon_elem[key]:
            elem_set.add(elem)
        process_step = process_step+1
        bar_achieve_label.update(process_step)
    bar_achieve_label.finish()
    elem_list = list(elem_set)
    elem_list.sort()
    
    with open('./data/elem_list.json', 'w') as f:
        json.dump(elem_list, f)

    with open('./data/label.json', 'w') as f:
        json.dump(data_label, f)

            
    db_cursor.close()
    connection.close()



if __name__ == '__main__':
    hostid = "192.168.1.101"
    host_user = "liningbo"
    host_passwd = "1a2a3a"
    database = "target"
    data_achieve(hostid, host_user, database, host_passwd)


    
