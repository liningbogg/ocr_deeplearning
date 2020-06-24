import pymysql
import json
import numpy as np

'''
ocr_data list process
for mnist likeness cnn
by liningbo

'''

def data_list(hostid, host_user, database, host_passwd, labels_contributor_set):
    
    """
    list polygon and its labels then convert them into train data for the mnist likeness model

    """
    elem_polygon_statistic = []
    connection = pymysql.connect(host=hostid, user=host_user, password=host_passwd, charset="utf8", use_unicode=True)
    db_cursor = connection.cursor()
    connection.select_db(database)
    sql_elem_achieve = 'select id, desc_info from ocr_chineseelem where create_user_id in (%s)' % ','.join(['%s'] * len(labels_contributor_set))
    db_cursor.execute(sql_elem_achieve, labels_contributor_set)
    query_elem = db_cursor.fetchall()
    for elem_id, desc_info in query_elem:
        sql_polygon_count = 'select COUNT(*) from ocr_polygonelem where elem_id = (%s)'
        db_cursor.execute(sql_polygon_count, elem_id)
        polygon_count_rs = db_cursor.fetchall()
        polygon_count = polygon_count_rs[0][0]
        elem_polygon_statistic.append({'id':elem_id, 'desc':desc_info, 'polygon_count':polygon_count})
    db_cursor.close()
    connection.close()

    elem_polygon_statistic = sorted(elem_polygon_statistic, key=lambda elem_polygon_statistic: elem_polygon_statistic['polygon_count'], reverse=True)
    with open("./data/elem_list.txt", "w") as f:
        for elem in elem_polygon_statistic:
            line = json.dumps(elem, ensure_ascii=False)
            line = line + "\n"
            f.write(line)
        


if __name__ == '__main__':
    hostid = "192.168.1.101"
    labels_contributor_set = ["pi"]
    host_user = "liningbo"
    host_passwd = "1a2a3a"
    database = "target"
    data_list(hostid, host_user, database, host_passwd, labels_contributor_set)
