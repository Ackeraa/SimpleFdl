import sys
from detection.frcnn_la import fasterrcnn_resnet50_fpn_feature
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from detection.retinanet_cal import retinanet_mobilenet, retinanet_resnet50_fpn_cal
from detection.train import *
import numpy as np
import torch
import random
import os
from config import *

class Server:
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.data_path = DATA_PATH
        self.model = MODEL
        self.gpu_ids = GPU_IDS
        self.total_epochs = TOTAL_EPOCHS
        self.total_rounds = TOTAL_ROUNDS
        self.client_nums = CLIENT_NUMS
        self.finished_clients = 0
        
        if 'voc2007' in self.dataset_name:
            dataset, self.num_classes = get_dataset(self.dataset, "trainval", get_transform(train=True), self.data_path)
        else:
            dataset, self.num_classes = get_dataset(self.dataset_name, "train", get_transform(train=True), self.data_path)
        self.img_nums = len(dataset)

        self.indices = []
        for _ in range(self.client_nums):
            indice = random.sample([_ for _ in range(self.img_nums)], int(self.img_nums * 0.9))
            self.indices.append(indice)

        if 'voc' in self.dataset_name:
            self.budget_num = 500
            if 'retina' in self.model:
                self.budget_num = 500
        else:
            self.budget_num = 1000

        if 'voc' in self.dataset_name:
            if 'faster' in self.model:
                self.model_path = os.path.join("modelp", "voc_faster.pt")
                if not self.load_model():
                    self.global_model = fasterrcnn_resnet50_fpn_feature(num_classes=self.num_classes, min_size=600, max_size=1000).state_dict()
                    self.save_model()
            elif 'retina' in args.model:
                self.model_path = os.path.join("modelp", "voc_retina.pt")
                if not self.load_model():
                    self.global_model = retinanet_resnet50_fpn_cal(num_classes=self.num_classes, min_size=600, max_size=1000).state_dict()
                    self.save_model()
        else:
            if 'faster' in self.model:
                self.model_path = os.path.join("modelp", "coco_faster.pt")
                if not self.load_model():
                    self.global_model = fasterrcnn_resnet50_fpn_feature(num_classes=self.num_classes, min_size=800, max_size=1333).state_dict()
                    self.save_model()
            elif 'retina' in self.model:
                self.model_path = os.path.join("modelp", "coco_retina.pt")
                if not self.load_model():
                    self.global_model = retinanet_resnet50_fpn_cal(num_classes=self.num_classes, min_size=800, max_size=1333).state_dict()
                    self.save_model()

    def load_model(self):
        try:
            self.global_model = torch.load(self.model_path)
        except FileNotFoundError:
            return False

        return True

    def save_model(self):
        torch.save(self.global_model, self.model_path)

    def load_p(self):
        path = os.path.join("modelp", "p.txt")
        if not os.path.exists(path):
            self.p = torch.tensor(np.random.random(self.num_classes), dtype=torch.float32)
            self.total_budget = 0
            self.save_p()
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                self.p = torch.tensor(list(map(float, lines[0].split())), dtype=torch.float32)
                self.total_budget = float(lines[1])

    def save_p(self):
        path = os.path.join("modelp", "p.txt")
        with open(path, 'w+') as f:
            for x in self.p.tolist():
                f.write(str(x) + " ")
            f.write("\n")
            f.write(str(self.total_budget))

    def get_p(self):
        self.load_p()
        if self.total_budget == 0:
            return self.p
        return torch.nn.functional.softmax(self.p/self.total_budget, -1)

    def update_p(self, q):
        self.get_p()
        self.p += q 
        self.total_budget += self.budget_num
        self.save_p()
     
    def aggregate(self, model):
        for layer in model:
            self.global_model[layer] += model[layer]

        self.finished_clients += 1
        if self.finished_clients == self.client_nums:
            self.finished_clients = 0
            for layer in model:
                self.global_model[layer] /= self.client_nums
            self.save_model()
        
        # for client in self.clients:
        #     client.task_model.load_state_dict(model)      

    def calculate_kl(self, log_qs):
        kls = []
        # p = torch.round(self.get_p()*10**3)/(10**3)
        # log_p = torch.round(p.log()*10**3)/(10**3)
        p = self.get_p()
        log_p = p.log()

        for log_q in log_qs:
            kl = 0
            for i in range(len(log_q)):
                kl += float(p[i]) * (float(log_p[i]) - log_q[i])
            kls.append(kl)
        
        return kls

