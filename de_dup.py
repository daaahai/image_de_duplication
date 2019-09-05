# -*- coding:utf-8 -*-

import os
import time
import copy
import sys
import types
import json
import shutil
import pickle
import imagehash
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DeDup(object):
    """
    Container of the de-duplication results
    """
    def __init__(self, json_list=[], img_list=[], hash_dict=dict(),
                 hash_df=pd.DataFrame(), dup_dict=dict(), dup_dir=[],
                 dup_image_list=[], dup_json_list=[]):
        self._json_list = json_list
        self._img_list = img_list
        self._hash_dict = hash_dict
        self._hash_df = hash_df
        self._dup_dict = dup_dict
        self._dup_dir = dup_dir
        self._dup_image_list = dup_image_list
        self._dup_json_list = dup_json_list

    # Getters and setters
    @property
    def json_list(self):
        return self._json_list

    @json_list.setter
    def json_list(self, json_list):
        self._json_list = json_list

    @property
    def img_list(self):
        return self._img_list

    @img_list.setter
    def img_list(self, img_list):
        self._img_list = img_list

    @property
    def hash_dict(self):
        return self._hash_dict

    @hash_dict.setter
    def hash_dict(self, hash_dict):
        self._hash_dict = hash_dict

    @property
    def hash_df(self):
        return self._hash_df

    @hash_df.setter
    def hash_df(self, hash_df):
        self._hash_df = hash_df

    @property
    def dup_dict(self):
        return self._dup_dict

    @dup_dict.setter
    def dup_dict(self, dup_dict):
        self._dup_dict = dup_dict

    @property
    def dup_dir(self):
        return self._dup_dir

    @dup_dir.setter
    def dup_dir(self, dup_dir):
        self._dup_dir = dup_dir

    @property
    def dup_image_list(self):
        return self._dup_image_list

    @dup_image_list.setter
    def dup_image_list(self, dup_image_list):
        self._dup_image_list = dup_image_list

    @property
    def dup_json_list(self):
        return self._dup_json_list

    @dup_json_list.setter
    def dup_json_list(self, dup_json_list):
        self._dup_json_list = dup_json_list

    @classmethod
    def load(cls, pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        # Set up object
        de_dup = cls(data['_json_list'], data['_img_list'], data['_hash_dict'], data['_hash_df'],
                     data['_dup_dict'], data['_dup_dir'], data['_dup_image_list'], data['_dup_json_list'])
        return de_dup

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(vars(self), f)


def get_img_list(img_dir, json_dir=None):
    """
    :param json_dir: 输入需要去重的json文件夹地址，需要该文件夹下所有json均为需要去重的图片对应json（如：'annotations/PP1/无水印'）
    :param img_dir: 输入上述json文件对应的图片地址，（如'images/PP1'），需要保证将json路径中的json_dir替换为img_dir时即找到对应图片
    :return: None, saved in DeDup class
    """
    if json_dir:
        for dirName, subdirList, fileList in os.walk(json_dir):
            for eachFile in fileList:
                # find json files
                json_path = os.path.join(dirName, eachFile)
                if os.path.splitext(json_path)[1] == '.json':
                    dup_res.json_list.append(json_path)
                # check extension
                img_path_ext = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path_ext.append(os.path.join(str(dirName).replace(json_dir, img_dir),
                                                     os.path.splitext(eachFile)[0] + ext))
                for img_path in img_path_ext:
                    if os.path.exists(img_path):
                        dup_res.img_list.append(img_path)
    else:
        for dirName, subdirList, fileList in os.walk(img_dir):
            for eachFile in fileList:
                # find img files
                img_path = os.path.join(dirName, eachFile)
                ext = ['.jpg', '.jpeg', '.png']
                if os.path.splitext(img_path)[1] in ext:
                    dup_res.img_list.append(img_path)


def get_hash_dict(hashsize):
    for img_path in dup_res.img_list:
        if os.path.exists(img_path):
            this_img_hash = img_hash(img_path, hashsize)
            if this_img_hash not in dup_res.hash_dict.keys():
                dup_res.hash_dict[this_img_hash] = [img_path]
            else:
                dup_res.hash_dict[this_img_hash].append(img_path)
            print('pHash dict length %d' % len(dup_res.hash_dict))


def img_hash(img_path, hash_size):
    return imagehash.phash(Image.open(img_path), hash_size=hash_size)


def get_dup_dict():
    n = 0
    for key in dup_res.hash_dict.keys():
        if len(dup_res.hash_dict[key]) > 1:
            dup_res.dup_dict[key] = dup_res.hash_dict[key]
            dup_res.dup_image_list += dup_res.hash_dict[key][1:]
            n += 1
    print('found %d pHash with more than 2 images' % n)


def copy_img(result_dict, result_dir):
    for key in list(result_dict.keys()):
        this_result_dir = os.path.join(result_dir, str(key))
        if not os.path.exists(this_result_dir):
            os.mkdir(this_result_dir)
        for img in result_dict[key]:
            shutil.copy(img, this_result_dir)


def knn_dup(threshold, k):
    for key in dup_res.hash_dict.keys():
        new_line = {'file': dup_res.hash_dict[key][0], 'hash': str(key)}
        dup_res.hash_df = dup_res.hash_df.append(new_line, ignore_index=True)
    lambdafunc = lambda x: pd.Series([int(i, 16) for key, i in zip(range(0, len(x['hash'])), x['hash'])])
    newcols = dup_res.hash_df.apply(lambdafunc, axis=1)
    newcols.columns = [str(i) for i in range(0, len(dup_res.hash_df.iloc[0]['hash']))]
    dup_res.hash_df = dup_res.hash_df.join(newcols)
    hash_str_len = len(str(dup_res.hash_df.get_value(0, 'hash')))
    print('calculating distances')
    t = KDTree(dup_res.hash_df[[str(i) for i in range(0, hash_str_len)]], metric='manhattan')
    distances, indices = t.query(dup_res.hash_df[[str(i) for i in range(0, hash_str_len)]], k=k)
    above_threshold_idx = np.argwhere((distances <= threshold) & (distances > 0))
    pairs_of_indexes_of_duplicate_images = set([tuple(sorted([indices[idx[0], 0], indices[idx[0], idx[1]]]))
                                                for idx in above_threshold_idx])
    return list(pairs_of_indexes_of_duplicate_images)


def pair_merge(pair_list):
    merged_list = copy.deepcopy(pair_list)
    length = len(pair_list)
    for i in range(length):
        for j in range(i):
            x = tuple(set(merged_list[i] + merged_list[j]))
            y = len(merged_list[j]) + len(merged_list[i])
            if len(x) < y:
                merged_list[i] = x
                merged_list[j] = ()
    return [merged_list[i] for i in range(len(merged_list)) if merged_list[i] != ()]


def img_2_json(img_dir, json_dir):  # 将图片路径转换为json路径
    for i_path in dup_res.dup_image_list:
        j_path = os.path.splitext(str(i_path).replace(img_dir, json_dir))[0] + '.json'
        dup_res.dup_json_list.append(j_path)


"""
参考代码：https://www.kaggle.com/kretes/duplicate-and-similar-images
去重逻辑：
    1.遍历所有文件，找到所有label的json文件，由于label和图片路径对应，通过json路径可以获得对应图片路径，只对有label的图片进行去重
    2.对所有图片进行perceptual hash，以hash值为key，存储在字典中
    3.遍历字典，找到其中一个哈希值对应不止一张图片的情况，删掉重复图片（此处筛选标准较为严格，要求pHash完全相同
    4.对于哈希值不同，但是图片内容实际相同的图片，再计算距离，用knn设定阈值筛选后合并去除

运行前需要修改：
    1.result_dir 输出结果的路径
    2.img_dir（必须）, json_dir (如果有，需要与img_dir对应）
    3.保存去重结果的文件，如：'de_duplication_pp.pkl'
"""

start_time = time.time()

# 如果是第一次去重
dup_res = DeDup()

# # 如果要基于之前的去重结果进行下一步去重则load之前储存的结果
# dup_res = DeDup.load('de_duplication_pp.pkl')

# 设定输出结果的路径
result_dir = '../images/de_duplication_pp'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# hashsize与图片fingerprint的resolution有关，越大则越清晰但计算时间变长。
hashsize = 16

# 遍历文件列表，获得hash_dict，如果是新增数据，则输入新数据的路径
for batch in ['PP1', 'PP2', 'PP3', 'PP4']:
    json_dir = '../annotations/' + batch + '/无水印'
    img_dir = '../images/' + batch
    get_img_list(img_dir, json_dir)  # 如果只对无标注图片去重，不需要输入json_dir

# # 如果只有图片，没有json
# img_dir = ''
# get_img_list(img_dir)

# 计算pHash并去重
get_hash_dict(hashsize)

# 将hash_dict中有两个以上元素的条目复制到dup_dict，同时会将重复图片写入dup_res.dup_image_list
get_dup_dict()

# 将上述运行后找到的图片复制到result_dir，可以人工看一下去重是否正确
copy_img(dup_res.dup_dict, result_dir)

# 在去除pHash完全相同的图片之后，用knn找近邻，需设定阈值
threshold = 100  # 设定判断图片相似的阈值
k = 5
# 哈希大小，阈值
unique_pairs = knn_dup(threshold, k)
dup_pairs = pair_merge(unique_pairs)

# 便于查看和复制图片
knn_d = dict()
for i in range(len(dup_pairs)):
    knn_d[i] = []
    for index in dup_pairs[i]:
        knn_d[i].append(dup_res.hash_df.iloc[index]['file'])

# 将此次筛选出来的重复图片加入dup_res.dup_image_list
for key in knn_d.keys():
    for path in knn_d[key][1:]:
        dup_res.dup_image_list.append(path)

# 将此次筛选出来的重复图片复制到文件夹中人工确认
copy_img(knn_d, result_dir)

# 确认无误后可以将img_path转换为json_path
for batch in ['PP1', 'PP2', 'PP3', 'PP4']:
    json_dir = '../annotations/' + batch + '/无水印'
    img_dir = '../images/' + batch
    img_2_json(img_dir, json_dir)

# 存储运行结果
dup_res.save(result_dir + 'de_duplication_pp.pkl')

print('Total time %.2f mins, %d / %d duplicate image found' %
      (((time.time()-start_time)/60), len(set(dup_res.dup_image_list)), len(dup_res.img_list)))
