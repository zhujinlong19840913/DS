from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import torch
import random
import numpy as np
import re
import os.path as osp
from collections import OrderedDict
import os
from operator import itemgetter
import pickle

def custom_sort(item):
    key, _ = item
    prefix = key.split('_')[0]  # 提取键的前缀部分，比如'0002'
    return int(prefix)

class dynamicSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, max_epochs):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.max_epochs = max_epochs
        self.num_reduce = 0  # 删减个数

        self.processed_pid_data = {}  # 身份与文件名映射字典，key：pid是身份顺序编码;value:真实身份，用于对指标文件中的身份进行转换
        index_dic = defaultdict(list)
        self.processed_img_data = {}
        for index, (name_path, pid, _, _, _, _) in enumerate(data_source):
            index_dic[pid].append(index)

            name = name_path.split('\\')[-1]
            self.processed_img_data[name] = index
            name = name.split('_')[0]
            name = ''.join(filter(str.isdigit, name))
            name = int(name) if name else 0  # 真实身份
            self.processed_pid_data.update({name: pid})
        self.pids = list(index_dic.keys())

        with open('scores/market/market_pid_scores.txt', 'r') as file:
            pids_score = file.readlines()

        self.pids_score_dict = OrderedDict()

        for item in pids_score:
            key, value = item.strip().split(':')
            key = self.processed_pid_data.get(int(key))
            self.pids_score_dict[key] = float(value)

        self.orderPids()

        self.image_dic = defaultdict(list)

        with open('scores/market/market_img_scores.txt', 'r') as file:
            imgs_score = file.readlines()

        self.imgs_score_dict = OrderedDict()

        for item in imgs_score:
            key, value = item.strip().split(':')
            self.imgs_score_dict[key] = float(value)

        self.orderImages()
        self.len = 0

    # 对构建有序的身份序列，用于训练遍历，即训练时身份的顺序
    def orderPids(self):
        # 根据value重排序字典
        self.pids_score_dict = dict(sorted(self.pids_score_dict.items(), key=itemgetter(1), reverse=True))  ##

        self.pids_order = list(self.pids_score_dict.keys())  # 获得排序好的pid

        assert len(self.pids) == len(self.pids_order), 'pids is not same to weight table.'  ##

        percentage = 0.7  # 重复系数
        num_add_pid = int(len(self.pids) * percentage)
        self.copied_list = self.pids_order[:num_add_pid]
        self.pids_order.extend(self.copied_list)

    def orderImages(self):
        self.imgs_score_dict = dict(sorted(self.imgs_score_dict.items(), key=itemgetter(1)))
        # 排序好的self.imgs_score_dict ,这样添加index的时候是按照权重大到小，字典self.image_dic中身份下的序列就是有序的，按从大到小
        for img_name in self.imgs_score_dict.keys():
            index = self.processed_img_data.get(img_name, None)  # 通过文件名获得对应的采样索引
            if index is not None:
                match = re.search(r'\d+', img_name)
                if match:
                    key = int(match.group())
                    key = self.processed_pid_data.get(int(key))
                    self.image_dic[key].append(index)

        self.len = 0
        for pid in self.image_dic.keys():
            img_idxs = self.image_dic[pid]
            num = len(img_idxs)

            # 计算需要补充的数量
            num_images_add = (self.num_instances - (num % self.num_instances)) % self.num_instances
            self.image_dic[pid].extend(self.image_dic[pid][:num_images_add])

    def update_weight(self, epoch, saved_data):
        self.pids_order.clear()
        self.image_dic.clear()
        new_pids_data = {}
        new_imgs_data = {}

        for values in saved_data.values():
            for item_weight in values:
                filename = item_weight[2]
                pattern_pid = re.compile(r'([-\d]+)_c(\d)')
                match_pid, _ = map(int, pattern_pid.search(filename).groups())

                value = float(item_weight[0])  # 去除多余的0
                if match_pid:
                    if match_pid not in new_pids_data:
                        new_pids_data[match_pid] = []
                    new_pids_data[match_pid].append(value)

                pattern_name = re.compile(r'([-\w]+\.jpg)')
                match_name = pattern_name.search(filename)
                if match_name:
                    image_name = match_name.group(1)
                    if image_name:
                        if image_name not in new_imgs_data:
                            new_imgs_data[image_name] = []
                        new_imgs_data[image_name].append(value)

        first = round(0.1 * ((self.max_epochs - epoch) / self.max_epochs), 4)
        second = 0.9
        # 计算每个身份对应值的平均数并保留4位小数
        new_pids_weights = {}
        for identity, values in new_pids_data.items():
            average_pids = sum(values) / len(values)
            identity = self.processed_pid_data.get(int(identity))
            if identity in self.pids_score_dict:
                pids_new_weight = first * self.pids_score_dict[identity] + second * round(average_pids, 4)
                new_pids_weights[identity] = round(pids_new_weight, 4)

        for key in self.pids_score_dict.keys():
            if key not in new_pids_weights:
                new_pids_weights[key] = self.pids_score_dict.get(key)

        # 计算每个图片对应值的平均数并保留4位小数
        new_imgs_weights = {}
        for image_name, values in new_imgs_data.items():
            average_imgs = sum(values) / len(values)
            if image_name in self.imgs_score_dict:
                imgs_new_weight = first * self.imgs_score_dict[image_name] + second * round(average_imgs, 4)
                new_imgs_weights[image_name] = round(imgs_new_weight, 4)

        if len(new_pids_weights) != len(self.pids_score_dict):
            for key in self.pids_score_dict:
                if key not in new_pids_data.keys():
                    new_imgs_weights[key] = self.pids_score_dict[key]

        self.pids_score_dict = new_pids_weights
        self.orderPids()

        if epoch > 80:
            self.num_reduce = round(len(self.pids) * 0.1)
            del self.pids_order[
                len(self.pids_order) - len(self.copied_list) - self.num_reduce:len(self.pids_order) - len(
                    self.copied_list)]

        for key, value in new_imgs_weights.items():
            str_imgs_key = str(key)
            if str_imgs_key in self.imgs_score_dict:
                self.imgs_score_dict[str_imgs_key] = round(value, 4)

        for key in self.imgs_score_dict.keys():
            if key not in new_imgs_weights.keys():
                new_imgs_weights[key] = self.imgs_score_dict.get(key)

        self.imgs_score_dict = new_imgs_weights
        self.orderImages()

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # 对数据集进行切块，按照self.num_instances
        for pid in self.pids:
            img_idxs = self.image_dic[pid]
            batch_idxs = []
            for idx in img_idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        pids_avai = self.pids_order[:len(self.pids) - self.num_reduce]  # 获取去重身份序列
        final_idxs = []
        copy_list = []

        while len(pids_avai) > self.num_pids_per_batch:
            temp_pids = pids_avai
            for pid in temp_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)

                if pid in self.copied_list:
                    copy_list.extend(batch_idxs.copy())

                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    pids_avai.remove(pid)
        copy_list.reverse()
        copy_list.extend(final_idxs)
        final_idxs = copy_list
        self.len = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.len

# New add by gu
class RandomIdentitySampler_IdUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances