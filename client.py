import time
import random
import math
import sys
import numpy as np
import math
import scipy.stats
import pickle

import torch
import torch.utils.data
from torch import nn, result_type
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms.functional as F

from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from cald.cald_helper import *
from ll4al.data.sampler import SubsetSequentialSampler
from detection.frcnn_la import fasterrcnn_resnet50_fpn_feature
from detection.retinanet_cal import retinanet_mobilenet, retinanet_resnet50_fpn_cal
# from phe import paillier

class Client:
    def __init__(self, gpu_id, data_path, dataset_name, model, indices, total_epochs, id):
        torch.cuda.set_device(gpu_id)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.id = id
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.model = model
        self.indices = indices
        self.total_epochs = total_epochs
        self.start_epoch = 0
        self.batch_size = 4
        self.workers = 0        
        self.aspect_ratio_group_factor = 3
        self.print_freq = 1000

        self.lr = 0.0025
        self.weight_decay = 1e-4
        self.lr_steps = [16, 9]
        self.lr_gamma = 0.1
        self.momentum = 0.9
        self.bp = 1.3 
        self.Sigma = 0.0
        self.mr = 1.2
        self.results_path = "results"

        self.augs = ['flip', 'cut_out', 'smaller_resize', 'rotation']

        print("Loading data")
        if 'voc2007' in self.dataset_name:
            self.dataset, self.num_classes = get_dataset(self.dataset_name, "trainval", get_transform(train=True), self.data_path)
            self.dataset_aug, _ = get_dataset(self.dataset_name, "trainval", None, self.data_path)
            self.dataset_test, _ = get_dataset(self.dataset_name, "test", get_transform(train=False), self.data_path)
        else:
            self.dataset, self.num_classes = get_dataset(self.dataset_name, "train", get_transform(train=True), self.data_path)
            self.dataset_aug, _ = get_dataset(self.dataset_name, "train", None, self.data_path)
            self.dataset_test, _ = get_dataset(self.dataset_name, "val", get_transform(train=False), self.data_path)

        print("Creating data loaders")
        if 'voc' in self.dataset_name:
            self.init_num = 500
            self.budget_num = 500
            if 'retina' in self.model:
                self.init_num = 1000
                self.budget_num = 500
        else:
            self.init_num = 5000
            self.budget_num = 1000

        random.shuffle(self.indices)
        self.labeled_set = self.indices[:self.init_num]
        self.unlabeled_set = list(set(self.indices) - set(self.labeled_set))
        self.train_sampler = SubsetRandomSampler(self.labeled_set)
        self.data_loader_test = DataLoader(self.dataset_test, batch_size=1, sampler=SequentialSampler(self.dataset_test),
                                    num_workers=self.workers, collate_fn=utils.collate_fn)

        print("Creating model")
        if 'voc' in self.dataset_name:
            if 'faster' in self.model:
                self.task_model = fasterrcnn_resnet50_fpn_feature(num_classes=self.num_classes, min_size=600, max_size=1000)
            elif 'retina' in self.model:
                self.task_model = retinanet_resnet50_fpn_cal(num_classes=self.num_classes, min_size=600, max_size=1000)
        else:
            if 'faster' in self.model:
                self.task_model = fasterrcnn_resnet50_fpn_feature(num_classes=self.num_classes, min_size=800, max_size=1333)
            elif 'retina' in self.model:
                self.task_model = retinanet_resnet50_fpn_cal(num_classes=self.num_classes, min_size=800, max_size=1333)

        self.cached_stamp = 0

    def wait(self):
        path = os.path.join("modelp", "global.pt")
        while True:
            stamp = os.stat(path).st_mtime
            if stamp != self.cached_stamp:
                self.cached_stamp - stamp
                break
            time.sleep(300)

    def load_model(self):
        path = os.path.join("modelp", "global.pt")
        self.task_model.load_state_dict(torch.load(path))

        self.task_model.to(self.device)

    def save_model(self):
        path = os.path.join("modelp", "client"+str(self.id)+".pt")
        torch.save(self.task_model.to(device='cpu').state_dict(), path) 

    def train(self, cycle):
        self.load_model()
        if self.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(self.dataset, k=self.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(self.train_sampler, group_ids, self.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(self.train_sampler, self.batch_size, drop_last=True)

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_sampler=train_batch_sampler, num_workers=self.workers,
                                                collate_fn=utils.collate_fn)

        params = [p for p in self.task_model.parameters() if p.requires_grad]
        self.task_optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.task_optimizer, milestones=self.lr_steps,
                                                                gamma=self.lr_gamma)
        
        # Start active learning cycles training
        print("Start training")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.total_epochs):
            self.task_model
            self.train_one_epoch(cycle, epoch)
            task_lr_scheduler.step()
            # evaluate after pre-set epoch
            '''
            if (epoch + 1) == self.total_epochs:
                voc_evaluate(self.task_model, self.data_loader_test, self.dataset_name, False, path=self.results_path)
                if 'coco' in self.dataset_name:
                    coco_evaluate(self.task_model, self.data_loader_test)
                elif 'voc' in self.dataset_name:
                    voc_evaluate(self.task_model, self.data_loader_test, self.dataset_name, False, path=self.results_path)
            '''

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        self.save_model()

    def train_one_epoch(self, cycle, epoch):
        self.task_model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, epoch)

        task_lr_scheduler = None

        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)
            task_lr_scheduler = utils.warmup_lr_scheduler(self.task_optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(self.data_loader, self.print_freq, header):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            task_loss_dict = self.task_model(images, targets)
            task_losses = sum(loss for loss in task_loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
            task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
            task_loss_value = task_losses_reduced.item()
            if not math.isfinite(task_loss_value):
                print("Loss is {}, stopping training".format(task_loss_value))
                print(task_loss_dict_reduced)
                sys.exit(1)

            self.task_optimizer.zero_grad()
            task_losses.backward()
            self.task_optimizer.step()
            if task_lr_scheduler is not None:
                task_lr_scheduler.step()
            metric_logger.update(task_loss=task_losses_reduced)
            metric_logger.update(task_lr=self.task_optimizer.param_groups[0]["lr"])
        return metric_logger

    def before_select(self):
        random.shuffle(self.unlabeled_set)

        if 'coco' in self.dataset_name:
            subset = self.unlabeled_set[:10000]
        else:
            subset = self.unlabeled_set

        print("Getting uncertainty")
        unlabeled_loader = DataLoader(self.dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                            num_workers=self.workers, pin_memory=True, collate_fn=utils.collate_fn)

        uncertainty, _cls_corrs = self.get_uncertainty(unlabeled_loader)
        print("Finished getting uncertainty")

        arg = np.argsort(np.array(uncertainty))

        # Update the labeled dataset and the unlabeled dataset, respectively
        self.cls_corrs_set = arg[:int(self.mr * self.budget_num)]

        cls_corrs = []
        for _ in range(int(self.budget_num * self.mr)):
            cls_corrs.append(np.random.random(self.num_classes))

        qs = torch.nn.functional.softmax(torch.tensor(cls_corrs), -1)

        self.log_qs = []
        self.cls_corrs = cls_corrs

        for q in qs:
            en_log_q = [x for x in q.log()]
            self.log_qs.append(en_log_q)

        return self.log_qs

    def select(self, kls):
        self.kls = kls

        tobe_labeled_set = np.argsort(np.array(self.kls))[-self.budget_num:]
        # Update the labeled dataset and the unlabeled dataset, respectively
        tobe_labeled_set = list(self.cls_corrs_set[tobe_labeled_set])
        self.labeled_set += tobe_labeled_set
        self.unlabeled_set = list(set(self.indices) - set(self.labeled_set))

        # Create a new dataloader for the updated labeled dataset
        self.train_sampler = SubsetRandomSampler(self.labeled_set)

        # Calculate q
        tobe_labeled_loader = DataLoader(self.dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(tobe_labeled_set),
                                            num_workers=self.workers, pin_memory=True, collate_fn=utils.collate_fn)
        result = []
        for _, targets in tobe_labeled_loader:
            for target in targets:
                cls_corr = [0] * self.cls_corrs[0].shape[0]
                for l in target['labels']:
                    cls_corr[l - 1] += 1
                result.append(cls_corr)
        self.q = np.sum(np.array(result), axis=0)

        return self.q

    def cls_kldiv(self, labeled_loader, cls_corrs):
        cls_inds = []
        result = []
        for _, targets in labeled_loader:
            for target in targets:
                cls_corr = [0] * cls_corrs[0].shape[0]
                for l in target['labels']:
                    cls_corr[l - 1] += 1
                result.append(cls_corr)
            # with open("vis/mutual_cald_label_{}_{}_{}_{}.txt".format(args.uniform, args.model, args.dataset, cycle),
            #           "wb") as fp:  # Pickling
            # pickle.dump(result, fp)
        for a in list(np.where(np.sum(cls_corrs, axis=1) == 0)[0]):
            cls_inds.append(a)
            # result.append(cls_corrs[a])
        while len(cls_inds) < self.budget_num:
            # batch cls_corrs together to accelerate calculating
            KLDivLoss = nn.KLDivLoss(reduction='none')
            _cls_corrs = torch.tensor(cls_corrs)
            _result = torch.tensor(np.mean(np.array(result), axis=0)).unsqueeze(0)
            p = torch.nn.functional.softmax(_result, -1)
            q = torch.nn.functional.softmax(_cls_corrs, -1)
            log_mean = ((p + q) / 2).log()
            jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2
            jsdiv[cls_inds] = -1
            max_ind = torch.argmax(jsdiv).item()
            cls_inds.append(max_ind)
            # result.append(cls_corrs[max_ind])
        return cls_inds

    def calcu_iou(self, A, B):
        '''
        calculate two box's iou
        '''
        width = min(A[2], B[2]) - max(A[0], B[0]) + 1
        height = min(A[3], B[3]) - max(A[1], B[1]) + 1
        if width <= 0 or height <= 0:
            return 0
        Aarea = (A[2] - A[0]) * (A[3] - A[1] + 1)
        Barea = (B[2] - B[0]) * (B[3] - B[1] + 1)
        iner_area = width * height
        return iner_area / (Aarea + Barea - iner_area)

    def get_uncertainty(self, unlabeled_loader):
        self.task_model.to(device=self.device)
        for aug in self.augs:
            if aug not in ['flip', 'multi_ga', 'color_adjust', 'color_swap', 'multi_color_adjust', 'multi_sp', 'cut_out',
                        'multi_cut_out', 'multi_resize', 'larger_resize', 'smaller_resize', 'rotation', 'ga', 'sp']:
                print('{} is not in the pre-set augmentations!'.format(aug))
        self.task_model.eval()
        with torch.no_grad():
            consistency_all = []
            mean_all = []
            cls_all = []
            for images, _ in unlabeled_loader:
                torch.cuda.synchronize()
                # only support 1 batch size
                aug_images = []
                aug_boxes = []
                for image in images:
                    output = self.task_model([F.to_tensor(image).cuda()])
                    ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = output[0]['boxes'], output[0][
                        'prob_max'], output[0]['scores_cls'], output[0]['labels'], output[0]['scores']
                    if len(ref_scores) > 40:
                        inds = np.round(np.linspace(0, len(ref_scores) - 1, 50)).astype(int)
                        ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = ref_boxes[inds], prob_max[
                            inds], ref_scores_cls[inds], ref_labels[inds], ref_scores[inds]
                    cls_corr = [0] * (self.num_classes - 1)
                    for s, l in zip(ref_scores, ref_labels):
                        cls_corr[l - 1] = max(cls_corr[l - 1], s.item())
                    cls_corrs = [cls_corr]
                    if output[0]['boxes'].shape[0] == 0:
                        consistency_all.append(0.0)
                        cls_all.append(np.mean(cls_corrs, axis=0))
                        break
                    # start augment
                    if 'flip' in self.augs:
                        flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                        aug_images.append(flip_image.cuda())
                        aug_boxes.append(flip_boxes.cuda())
                    if 'ga' in self.augs:
                        ga_image = GaussianNoise(image, 16)
                        aug_images.append(ga_image.cuda())
                        aug_boxes.append(ref_boxes.cuda())
                    if 'multi_ga' in self.augs:
                        for i in range(1, 7):
                            ga_image = GaussianNoise(image, i * 8)
                            aug_images.append(ga_image.cuda())
                            aug_boxes.append(ref_boxes.cuda())
                    if 'color_adjust' in self.augs:
                        color_adjust_image = ColorAdjust(image, 1.5)
                        aug_images.append(color_adjust_image.cuda())
                        aug_boxes.append(ref_boxes)
                    if 'color_swap' in self.augs:
                        color_swap_image = ColorSwap(image)
                        aug_images.append(color_swap_image.cuda())
                        aug_boxes.append(ref_boxes)
                    if 'multi_color_adjust' in self.augs:
                        for i in range(2, 6):
                            color_adjust_image = ColorAdjust(image, i)
                            aug_images.append(color_adjust_image.cuda())
                            aug_boxes.append(reference_boxes)
                    if 'sp' in self.augs:
                        sp_image = SaltPepperNoise(image, 0.1)
                        aug_images.append(sp_image.cuda())
                        aug_boxes.append(ref_boxes)
                    if 'multi_sp' in self.augs:
                        for i in range(1, 7):
                            sp_image = SaltPepperNoise(image, i * 0.05)
                            aug_images.append(sp_image.cuda())
                            aug_boxes.append(ref_boxes)
                    if 'cut_out' in self.augs:
                        cutout_image = cutout(image, ref_boxes, ref_labels, 2)
                        aug_images.append(cutout_image.cuda())
                        aug_boxes.append(ref_boxes)
                    if 'multi_cut_out' in self.augs:
                        for i in range(1, 5):
                            cutout_image = cutout(image, ref_boxes, ref_labels, i)
                            aug_images.append(cutout_image.cuda())
                            aug_boxes.append(ref_boxes)
                    if 'multi_resize' in self.augs:
                        for i in range(7, 10):
                            resize_image, resize_boxes = resize(image, ref_boxes, i * 0.1)
                            aug_images.append(resize_image.cuda())
                            aug_boxes.append(resize_boxes)
                    if 'larger_resize' in self.augs:
                        resize_image, resize_boxes = resize(image, ref_boxes, 1.2)
                        aug_images.append(resize_image.cuda())
                        aug_boxes.append(resize_boxes)
                    if 'smaller_resize' in self.augs:
                        resize_image, resize_boxes = resize(image, ref_boxes, 0.8)
                        aug_images.append(resize_image.cuda())
                        aug_boxes.append(resize_boxes)
                    if 'rotation' in self.augs:
                        rot_image, rot_boxes = rotate(image, ref_boxes, 5)
                        aug_images.append(rot_image.cuda())
                        aug_boxes.append(rot_boxes)
                    outputs = []
                    for aug_image in aug_images:
                        outputs.append(self.task_model([aug_image])[0])
                    consistency_aug = []
                    mean_aug = []
                    for output, aug_box, aug_image in zip(outputs, aug_boxes, aug_images):
                        consistency_img = 1.0
                        mean_img = []
                        boxes, scores_cls, pm, labels, scores = output['boxes'], output['scores_cls'], output['prob_max'], \
                                                                output['labels'], output['scores']
                        cls_corr = [0] * (self.num_classes - 1)
                        for s, l in zip(scores, labels):
                            cls_corr[l - 1] = max(cls_corr[l - 1], s.item())
                        cls_corrs.append(cls_corr)
                        if len(boxes) == 0:
                            consistency_aug.append(0.0)
                            mean_aug.append(0.0)
                            continue
                        for ab, ref_score_cls, ref_pm, ref_score in zip(aug_box, ref_scores_cls, prob_max, ref_scores):
                            width = torch.min(ab[2], boxes[:, 2]) - torch.max(ab[0], boxes[:, 0])
                            height = torch.min(ab[3], boxes[:, 3]) - torch.max(ab[1], boxes[:, 1])
                            Aarea = (ab[2] - ab[0]) * (ab[3] - ab[1])
                            Barea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                            iner_area = width * height
                            iou = iner_area / (Aarea + Barea - iner_area)
                            iou[width < 0] = 0.0
                            iou[height < 0] = 0.0
                            p = ref_score_cls.cpu().numpy()
                            q = scores_cls[torch.argmax(iou)].cpu().numpy()
                            m = (p + q) / 2
                            js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
                            if js < 0:
                                js = 0
                            # consistency_img.append(torch.abs(
                            #     torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - self.bp).item())
                            consistency_img = min(consistency_img, torch.abs(
                                torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - self.bp).item())
                            mean_img.append(torch.abs(
                                torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                        consistency_aug.append(np.mean(consistency_img))
                        mean_aug.append(np.mean(mean_img))
                    consistency_all.append(np.mean(consistency_aug))
                    mean_all.append(mean_aug)
                    cls_corrs = np.mean(np.array(cls_corrs), axis=0)
                    cls_all.append(cls_corrs)
        mean_aug = np.mean(mean_all, axis=0)
        print(mean_aug)
        return consistency_all, cls_all

if __name__ == "__main__":
    client = Client()
    #client.train()
