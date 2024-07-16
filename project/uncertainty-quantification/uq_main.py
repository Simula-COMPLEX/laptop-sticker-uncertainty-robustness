#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.01.2024 15:21
# @Author  : Chengjie
# @File    : detection_offline.py
# @Software: PyCharm
import json
import os
import time

import torchmetrics
from torch import nn
from tqdm import tqdm
import cv2 as cv
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
import torch
from uq_metrics.metrics import UQMetrics
from utils import load_camera_calibration, load_model, run_model, calc_3d_point
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from uq_metrics.utils import cluster, DBSCANCluster, get_kdist_plot
from dataset_def_pl import StickerData

RUNTIME_TYPE = 'normal'  # Choices == 'onnx' and 'normal'
LABELS = {'Background': 0, 'Logo': 1, 'Sticker': 2}


class Detection:
    def __init__(self, images_path, model_name, checkout_path, dropout):
        self.model = load_model(RUNTIME_TYPE, model_name=model_name, checkout_path=checkout_path)
        if model_name in ['ssd300_vgg16', 'ssdlite320_mobilenet_v3_large']:
            self.model.model.head.dropout = nn.Dropout(p=dropout)
        else:
            self.model.model.backbone.fpn.dropout = nn.Dropout(p=dropout)

        self.p, self.d, self.dist_maps = load_camera_calibration()
        self.min_score = 0.6
        self.show_bounding_box_sticker = True
        self.show_bounding_box_logo = True
        self.uq = UQMetrics()

        self.images = os.listdir(os.path.join(images_path))
        self.images = [images_path + '/' + image for image in self.images]
        self.images = [file for file in self.images if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        with open(images_path + '/annotations.coco.json') as f:
            self.annotations = json.load(f)

    def get_target(self, path):
        image_name = path.split('/')[-1]

        # get the image id and annotations for the image
        image_id = [self.annotations['images'][i]['id'] for i in range(len(self.annotations['images'])) if
                    self.annotations['images'][i]['file_name'] == image_name]
        if len(image_id) != 1:
            print(image_name)
        assert len(image_id) == 1
        image_id = image_id[0]

        annotations = [annotation for annotation in self.annotations['annotations'] if
                       annotation['image_id'] == image_id]

        boxes = []
        areas = []
        labels = []
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            area = annotation['area']
            label = annotation['category_id']
            areas.append(area)
            boxes.append([x, y, x + w, y + h])
            labels.append(label)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id,
                  'area': areas,
                  'iscrowd': iscrowd,
                  'image_name': image_name}
        return target

    def process_images(self, path="test_images/IMG_3774.jpg"):
        image_og = cv.imread(path)
        # Undistorts images with the maps calculated in load_camera_calibration()
        image_og = cv.remap(image_og, self.dist_maps[0], self.dist_maps[1], cv.INTER_LINEAR)
        scale = 1.0
        image_rz = cv.resize(image_og, (0, 0), fx=scale, fy=scale)

        try:
            targets = self.get_target(path)

        except:
            targets = None
        return image_og, image_rz, targets

    def predict(self, image_rz):
        preds = run_model(image_rz, self.model, RUNTIME_TYPE)

        for j in range(len(preds[0]['scores']) - 1, -1, -1):
            if preds[0]['scores'][j] < self.min_score or preds[0]['labels'][j] == 0 or (
                    preds[0]['labels'][j] == 1 and not self.show_bounding_box_logo):
                preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j + 1:]))
                preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j + 1:]))
                preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j + 1:]))
                preds[0]['logits'] = torch.cat((preds[0]['logits'][:j], preds[0]['logits'][j + 1:]))

        for pred in preds:
            for key, value in pred.items():
                pred[key] = value.cpu()

        boxes = preds[0]['boxes'].cpu().detach().numpy()
        labels = preds[0]['labels'].cpu().detach().numpy()
        scores = preds[0]['scores'].cpu().detach().numpy()
        logits = preds[0]['logits'].cpu().detach().numpy()

        return preds, boxes, labels, scores, logits

    def draw_boxes(self, boxes, labels, scores, image_og):
        scale = 1.0
        boxes = boxes / scale

        center_3d_points = []
        labels = [key for value in labels for key, val in LABELS.items() if val == value]

        for box, label, score in zip(boxes, labels, scores):
            if self.show_bounding_box_sticker:
                if label == 'Sticker':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)
                    image_og = cv.putText(image_og, str(label), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str, center_3d = calc_3d_point(box, self.p)
                    # center_3d_points.append(center_3d)
                    center_3d_points.append([x, y])

                    # draw
                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)

            if self.show_bounding_box_logo:
                if label == 'Logo':
                    image_og = cv.rectangle(image_og, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                            (255, 0, 0), 2)
                    image_og = cv.putText(image_og, str(label), (int(box[0]), int(box[1])),
                                          cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                                          cv.LINE_AA)  # (255,0,0) is blue

                    x = int((box[0] + box[2]) / 2)
                    y = int((box[1] + box[3]) / 2)
                    center_3d_str, center_3d = calc_3d_point(box, self.p)
                    # center_3d_points.append(center_3d)
                    center_3d_points.append([x, y])

                    image_og = cv.circle(image_og, (int(x), int(y)), 3, (0, 0, 255), -1)

        # cv.imwrite('./image_labeled/test_{}.png'.format(time.time()), image_og)

        return center_3d_points

    @staticmethod
    def put_text(text, box, image):
        cv.putText(image, text, (int(box[0] - 260), int(box[1])),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                   cv.LINE_AA)  # (255,0,0) is blue

    @staticmethod
    def avg_uq(output):
        vr, entropy, mi, tv_box, ps = [], [], [], [], []
        for key in output.keys():
            vr.append(output[key]['variation_ratio [classification]'])
            entropy.append(output[key]['entropy [classification]'])
            mi.append(output[key]['mutual_info [classification]'])
            tv_box.append(output[key]['total_var_box [regression]'])
            ps.append(output[key]['predictive_surface [regression]'])

        # print(np.array(entropy).mean())

        output.update({'Metrics_Avg': {
            'variation_ratio [classification]': np.array(vr).mean(),
            'entropy [classification]': np.array(entropy).mean(),
            'mutual_info [classification]': np.array(mi).mean(),
            'total_var_box [regression]': np.array(tv_box).mean(),
            'predictive_surface [regression]': np.array(ps).mean(),
        },

        })
        return output

    def predict_multi(self, image_rz, image_og, image_n, n=10):
        predictions = {}
        pred_id = 0

        mc_locations = []

        for i in range(n):
            preds, boxes, labels, scores, logits = self.predict(image_rz)
            center_points = self.draw_boxes(boxes, labels, scores, image_og)
            for box, label, score, logit, center_point in zip(boxes, labels, scores, logits, center_points):
                center_point = np.array(center_point)
                mc_locations.append(np.concatenate((box, center_point), axis=None))

                predictions.update({'label_{}'.format(pred_id):
                    {
                        'box': box.tolist(),
                        'label': label.tolist(),
                        'score': score.tolist(),
                        'logit': logit.tolist(),
                        # 'center_point': center_point
                    }
                })
                pred_id += 1

        # k = 2 * 4 - 1
        # get_kdist_plot(mc_locations, k)

        dbscan_cluster = DBSCANCluster(x=mc_locations)
        predictions = dbscan_cluster.cluster_preds(preds=predictions)
        # print(json.dumps(predictions, indent=4))
        # print(dbscan_cluster.cluster_labels)
        # cluster(mc_locations)

        for key in predictions.keys():
            # self.put_text(key, predictions[key]['box'][0], image_og)
            logit_sample_trans = np.transpose(predictions[key]['logit'])
            vr = self.uq.cal_vr(predictions[key]['logit'])
            shannon_entropy = self.uq.calcu_entropy(np.mean(logit_sample_trans, axis=1))
            mi = self.uq.calcu_mi(predictions[key]['logit'])
            tv_box = self.uq.calcu_tv_2(predictions[key]['box'], tag='bounding_box')
            predictive_surface = self.uq.calcu_prediction_surface(predictions[key]['box'])

            print(self.uq.hull[0].simplices)
            # for i in range(len(predictions[key]['box'])):
            #     self.draw_boxes(np.asarray([predictions[key]['box'][i]]), [predictions[key]['label'][i]],
            #                     [predictions[key]['score'][i]], image_og)

            predictions[key].update({
                'detection times (out of {})'.format(n): len(predictions[key]['score']),
                'variation_ratio [classification]': vr,
                'entropy [classification]': shannon_entropy,
                'mutual_info [classification]': mi,
                'total_var_box [regression]': tv_box,
                'predictive_surface [regression]': predictive_surface
            })

        predictions = self.avg_uq(predictions)

        timestamp = str(int(time.time()))
        tag = image_n
        with open('./image_labeled/output_{}_{}.json'.format(tag, timestamp), 'w') as f:
            json.dump(predictions, f, indent=4)
        cv.imwrite('./image_labeled/multi_boxes_{}_{}.png'.format(tag, timestamp), image_og)

    def predict_multi_draw(self, image_rz, image_og, n=10):
        for i in range(n):
            preds, boxes, labels, scores, logits = self.predict(image_rz)
            self.draw_boxes(boxes, labels, scores, image_og)
        cv.imwrite('./image_labeled/multi_boxes_{}.png'.format(time.time()), image_og)


def run_single(model_name='retinanet_resnet50_fpn',
               checkout_path='checkpoints/retinanet_resnet50_fpn/epoch=30-step=7471.ckpt'):
    i_n = 'image_stable_diffusion_97'
    detector = Detection('data/test', model_name=model_name, checkout_path=checkout_path, dropout=0.9)
    i_og, i_rz, target = detector.process_images(path="dataset/sdimg/adv_run_1/{}.jpg".format(i_n))
    # detector.predict_multi_draw(i_rz, i_og, n=1)
    detector.predict_multi(i_rz, i_og, i_n, n=10)


def run(model_name, checkout_path):
    map_metric = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100])
    iou_metric = torchmetrics.detection.IntersectionOverUnion()
    detector = Detection('data/test', model_name=model_name, checkout_path=checkout_path)
    sticker_data = StickerData(train_folder='data/train', valid_folder='data/valid', test_folder='data/val')
    sticker_data.setup(stage='test')

    dataloader = sticker_data.test_dataloader()

    with torch.no_grad():
        for i, (image, target, i_name) in tqdm(enumerate(dataloader), total=153):
            torch.cuda.empty_cache()
            image = list(image)
            image[0] = image[0]
            image = tuple(image)

            preds = detector.model(image)

            for j in range(len(preds[0]['scores']) - 1, -1, -1):
                if preds[0]['scores'][j] < 0.6 or preds[0]['labels'][j] != 2:
                    preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j + 1:]))
                    preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j + 1:]))
                    preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j + 1:]))
                    # preds[0]['logits'] = torch.cat((preds[0]['logits'][:j], preds[0]['logits'][j + 1:]))

            for j in range(len(target[0]['labels']) - 1, -1, -1):
                if target[0]['labels'][j] != 2:
                    target[0]['boxes'] = torch.cat((target[0]['boxes'][:j], target[0]['boxes'][j + 1:]))
                    target[0]['labels'] = torch.cat((target[0]['labels'][:j], target[0]['labels'][j + 1:]))
                    target[0]['image_id'] = torch.cat((target[0]['image_id'][:j], target[0]['image_id'][j + 1:]))

            # move the preds and target to cpu
            for pred in preds:
                for key, value in pred.items():
                    pred[key] = value.cpu()

            map_metric.update(preds, target)
            iou_metric.update(preds, target)

        map_results = map_metric.compute()
        iou_results = iou_metric.compute()

        mean_iou = []
        for iou_matrix in iou_metric.iou_matrix:
            try:
                if iou_matrix.nelement() == 0:
                    continue
                mean = torch.mean(torch.max(iou_matrix, dim=1).values)
                mean_iou.append(mean.item())
            except:
                pass

        print(map_results)
        print(f'IoU: {iou_results["iou"]}, Mean_IoU: {sum(mean_iou) / len(mean_iou)}')
        print(f'mAP: {map_results["map"]}, mAP_50: {map_results["map_50"]}, mAP_75: {map_results["map_75"]}')
        # pd.DataFrame([[model_name, sum(mean_iou) / len(mean_iou), map_results["map"].item()]]). \
        #     to_csv('./performance.csv', mode='a', header=False, index=False)


def run_uq(model_name, checkout_path, save_path, uq_logs, dropout, dataset='val', T=40):
    map_metric_overall = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100],
                                                                     iou_thresholds=[0.5])
    detector = Detection('data/test', model_name=model_name, checkout_path=checkout_path, dropout=dropout)
    sticker_data = StickerData(train_folder='data/train', valid_folder='data/val',
                               test_folder='dataset/{}'.format(dataset))
    sticker_data.setup(stage='test')

    dataloader = sticker_data.test_dataloader()

    uq_metrics = {'vr': [], 'ie': [], 'mi': [], 'tr': [], 'ps': []}
    with torch.no_grad():
        for i, (image, target, i_name) in tqdm(enumerate(dataloader), total=len(dataloader)):
            torch.cuda.empty_cache()
            image = list(image)
            image[0] = image[0]
            image = tuple(image)

            predictions, pred_id, mc_locations = {}, 0, []
            #
            map_metric = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100],
                                                                     iou_thresholds=[0.5])
            for t in range(T):

                preds = detector.model(image)

                for j in range(len(preds[0]['scores']) - 1, -1, -1):
                    if preds[0]['scores'][j] < 0.6 or preds[0]['labels'][j] not in [2]:
                        preds[0]['boxes'] = torch.cat((preds[0]['boxes'][:j], preds[0]['boxes'][j + 1:]))
                        preds[0]['labels'] = torch.cat((preds[0]['labels'][:j], preds[0]['labels'][j + 1:]))
                        preds[0]['scores'] = torch.cat((preds[0]['scores'][:j], preds[0]['scores'][j + 1:]))
                        preds[0]['logits'] = torch.cat((preds[0]['logits'][:j], preds[0]['logits'][j + 1:]))

                for j in range(len(target[0]['labels']) - 1, -1, -1):
                    if target[0]['labels'][j] not in [2]:
                        target[0]['boxes'] = torch.cat((target[0]['boxes'][:j], target[0]['boxes'][j + 1:]))
                        target[0]['labels'] = torch.cat((target[0]['labels'][:j], target[0]['labels'][j + 1:]))
                        target[0]['image_id'] = torch.cat((target[0]['image_id'][:j], target[0]['image_id'][j + 1:]))

                # move the preds and target to cpu
                for pred in preds:
                    for key, value in pred.items():
                        pred[key] = value.cpu()

                boxes = preds[0]['boxes'].cpu().detach().numpy()
                labels = preds[0]['labels'].cpu().detach().numpy()
                scores = preds[0]['scores'].cpu().detach().numpy()
                logits = preds[0]['logits'].cpu().detach().numpy()

                for box, label, score, logit in zip(boxes, labels, scores, logits):
                    mc_locations.append(
                        np.concatenate((box, np.array([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])),
                                       axis=None))
                    predictions.update({'label_{}'.format(pred_id):
                        {
                            'box': box.tolist(),
                            'label': label.tolist(),
                            'score': score.tolist(),
                            'logit': logit.tolist(),
                        }
                    })
                    pred_id += 1

                map_metric.update(preds, target)
                map_metric_overall.update(preds, target)

            if len(mc_locations) <= 1:
                continue
            dbscan_cluster = DBSCANCluster(x=mc_locations)
            predictions = dbscan_cluster.cluster_preds(preds=predictions)
            #
            for key in predictions.keys():
                logit_sample_trans = np.transpose(predictions[key]['logit'])
                vr = detector.uq.cal_vr(predictions[key]['logit'])
                shannon_entropy = detector.uq.calcu_entropy(np.mean(logit_sample_trans, axis=1))
                mi = detector.uq.calcu_mi(predictions[key]['logit'])
                tv_box = detector.uq.calcu_tv_2(predictions[key]['box'], tag='bounding_box')
                predictive_surface = detector.uq.calcu_prediction_surface(predictions[key]['box'])

                predictions[key].update({
                    'detection times (out of {})'.format(T): len(predictions[key]['score']),
                    'variation_ratio [classification]': vr,
                    'entropy [classification]': shannon_entropy,
                    'mutual_info [classification]': mi,
                    'total_var_box [regression]': tv_box,
                    'predictive_surface [regression]': predictive_surface
                })
            predictions = detector.avg_uq(predictions)

            with open('{}/dropout_{}_{}.json'.format(uq_logs, dropout, i_name[0].split('/')[-1].split('.')[0]),
                      'w') as f:
                json.dump(predictions, f, indent=4)

            map_results = map_metric.compute()

            pd.DataFrame([[i, i_name[0], map_results["map"].item(),
                           predictions['Metrics_Avg']['variation_ratio [classification]'],
                           predictions['Metrics_Avg']['entropy [classification]'],
                           predictions['Metrics_Avg']['mutual_info [classification]'],
                           predictions['Metrics_Avg']['total_var_box [regression]'],
                           predictions['Metrics_Avg']['predictive_surface [regression]'],
                           map_results["map_50"].item(),
                           map_results["map_75"].item(),
                           map_results["map_small"].item(),
                           map_results["map_medium"].item(),
                           map_results["map_large"].item(),
                           map_results["mar_1"].item(),
                           map_results["mar_5"].item(),
                           map_results["mar_100"].item(),
                           map_results["mar_small"].item(),
                           map_results["mar_medium"].item(),
                           map_results["mar_large"].item()
                           ]]). \
                to_csv(save_path, mode='a', header=False, index=False)

            uq_metrics['vr'].append(predictions['Metrics_Avg']['variation_ratio [classification]'])
            uq_metrics['ie'].append(predictions['Metrics_Avg']['entropy [classification]'])
            uq_metrics['mi'].append(predictions['Metrics_Avg']['mutual_info [classification]'])
            uq_metrics['tr'].append(predictions['Metrics_Avg']['total_var_box [regression]'])
            uq_metrics['ps'].append(predictions['Metrics_Avg']['predictive_surface [regression]'])

        map_results_all = map_metric_overall.compute()

        pd.DataFrame([[len(dataloader), 'overall', map_results_all['map'].item(),
                       sum(uq_metrics['vr']) / len(uq_metrics['vr']),
                       sum(uq_metrics['ie']) / len(uq_metrics['ie']),
                       sum(uq_metrics['mi']) / len(uq_metrics['mi']),
                       sum(uq_metrics['tr']) / len(uq_metrics['tr']),
                       sum(uq_metrics['ps']) / len(uq_metrics['ps']),
                       map_results["map_50"].item(),
                       map_results["map_75"].item(),
                       map_results["map_small"].item(),
                       map_results["map_medium"].item(),
                       map_results["map_large"].item(),
                       map_results["mar_1"].item(),
                       map_results["mar_5"].item(),
                       map_results["mar_100"].item(),
                       map_results["mar_small"].item(),
                       map_results["mar_medium"].item(),
                       map_results["mar_large"].item()
                       ]]). \
            to_csv(save_path, mode='a', header=False, index=False)


if __name__ == '__main__':
    models = {
        'retinanet_resnet50_fpn': 'checkpoints/retinanet_resnet50_fpn/epoch=30-step=7471.ckpt',
        'retinanet_resnet50_fpn_v2': 'checkpoints/retinanet_resnet50_fpn_v2/epoch=31-step=7712.ckpt',
        'fasterrcnn_resnet50_fpn': 'checkpoints/fasterrcnn_resnet50_fpn/epoch=14-step=1815.ckpt',
        'fasterrcnn_resnet50_fpn_v2': 'checkpoints/fasterrcnn_resnet50_fpn_v2/epoch=16-step=8177.ckpt',
        'ssd300_vgg16': 'checkpoints/ssd300_vgg16/epoch=33-step=4114.ckpt',
        'ssdlite320_mobilenet_v3_large': 'checkpoints/ssdlite320_mobilenet_v3_large/epoch=18-step=9139.ckpt'
    }

    for model_n in models.keys():
        for d_i in range(0, 11):
            dataset_n = 'sdimg/org' if d_i == 0 else 'sdimg/adv_run_{}'.format(d_i)

            for drop_out in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                print(model_n, dataset_n, drop_out)
                headers = ['image', 'image_name', 'MAP', 'UQ[VR]', 'UQ[IE]', 'UQ[MI]', 'UQ[TR]', 'UQ[PS]',
                           'MAP[50]', 'MAP[75]', 'MAP[Small]', 'MAP[Medium]', 'MAP[Large]', 'MAR[1]', 'MAR[5]',
                           'MAR[100]', 'MAR[Small]', 'MAR[Medium]', 'MAR[Large]']

                logs = 'experiment_results/{}/dataset/{}'.format(model_n, dataset_n)
                if not os.path.exists(logs):
                    os.makedirs(logs)

                f_n = './experiment_results/{}/logs_{}_{}.csv'.format(model_n, drop_out, dataset_n.replace('/', '-'))
                pd.DataFrame([headers]).to_csv(f_n, mode='w', header=False, index=False)
                run_uq(model_name=model_n, checkout_path=models[model_n], T=20, dataset=dataset_n,
                       save_path=f_n, uq_logs=logs, dropout=drop_out)

    # for i in range(1):
    #     run_single(model_name='retinanet_resnet50_fpn_v2',
    #                checkout_path='checkpoints/retinanet_resnet50_fpn_v2/epoch=31-step=7712.ckpt')
