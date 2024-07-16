#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/03/2024 21:45
# @Author  : Chengjie
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import torch
from scipy.spatial import ConvexHull


class UQMetrics:
    def __init__(self):
        self.variation_ratio = 0
        self.shannon_entropy = 0
        self.mutual_information = 0
        self.total_var_center_point = 0
        self.total_var_bounding_box = 0
        self.prediction_surface = -1
        self.hull = []
        self.box = []
        # pass

    def cal_vr(self, events):
        """
        Variation Ratio Calculation
        :return:
        """
        y = torch.argmax(torch.Tensor(events), dim=1).numpy()
        unique, counts = np.unique(y, return_counts=True)
        self.variation_ratio = 1 - np.max(counts) / np.sum(counts)
        return self.variation_ratio

    def calcu_entropy(self, events, ets=1e-15, base=2):
        """
        Shannon Entropy Calculation
        :param events:
        :param ets:
        :param base:
        :return:
        """
        self.shannon_entropy = round(-sum([p * (np.log(p + ets) / np.log(base)) for p in events]), 5)
        return self.shannon_entropy

    def calcu_mi(self, events, ets=1e-15, base=2):
        def calcu(e):
            t = 0
            for s in e:
                t += sum([p * (np.log(p + ets) / np.log(base)) for p in s])
            return t / len(e)

        self.mutual_information = self.calcu_entropy(events=np.mean(np.transpose(events), axis=1)) + calcu(events)
        return self.mutual_information

    # @staticmethod
    def calcu_tv(self, matrix, tag):
        """
        calculate total variance for a multi-dimensional matrix
        :param matrix:
        :param tag:
        :return: total variance
        """
        trans = np.array(matrix).T
        cov_matrix = np.cov(trans)
        if tag == 'bounding_box':
            self.total_var_bounding_box = np.trace(cov_matrix)
            return self.total_var_bounding_box
        elif tag == 'center_point':
            self.total_var_center_point = np.trace(cov_matrix)
            return self.total_var_center_point

    def calcu_tv_2(self, matrix, tag):
        """
        calculate total variance for a multi-dimensional matrix
        :param matrix:
        :param tag:
        :return: total variance
        """
        import statistics

        self.total_var_bounding_box = 0
        trans = np.array(matrix).T

        if tag == 'bounding_box':
            for m in trans:
                self.total_var_bounding_box += np.var(m)

            # if self.total_var_bounding_box > 2000:
            #     print(trans)
            return self.total_var_bounding_box

    def calcu_mutual_information(self, X, Y, Z):
        """
        Calculate mutual information between three discrete random variables X, Y, and Z.
        http://www.scholarpedia.org/article/Mutual_information#:~:text=Mutual%20information%20is%20one%20of,variable%20given%20knowledge%20of%20another.
        Parameters:
            X, Y, Z : array-like, shape (n_samples,)
                Arrays containing discrete random variables.

        Returns:
            mutual_info : float
                The mutual information between X, Y, and Z.
        """
        # Compute joint probability distribution
        unique_X = np.unique(X)
        unique_Y = np.unique(Y)
        unique_Z = np.unique(Z)

        joint_probs = np.zeros((len(unique_X), len(unique_Y), len(unique_Z)))
        for i, x in enumerate(unique_X):
            for j, y in enumerate(unique_Y):
                for k, z in enumerate(unique_Z):
                    joint_probs[i, j, k] = np.sum(np.logical_and(np.logical_and(X == x, Y == y), Z == z)) / float(
                        len(X))

        # Compute marginal probability distributions
        px = np.sum(joint_probs, axis=(1, 2))
        py = np.sum(joint_probs, axis=(0, 2))
        pz = np.sum(joint_probs, axis=(0, 1))

        # Compute mutual information
        mutual_info = 0.0
        for i, x in enumerate(unique_X):
            for j, y in enumerate(unique_Y):
                for k, z in enumerate(unique_Z):
                    if joint_probs[i, j, k] > 0.0:
                        mutual_info += joint_probs[i, j, k] * np.log2(joint_probs[i, j, k] / (px[i] * py[j] * pz[k]))

        self.mutual_information = mutual_info
        return self.mutual_information

    # Example usage:
    # X, Y, and Z are arrays containing discrete random variables
    # mutual_info_score = mutual_information(X, Y, Z)

    def calcu_prediction_surface(self, boxes):
        cluster_df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
        self.prediction_surface = -1
        self.hull = []
        sf_tmp = 0
        if cluster_df.shape[0] > 2:
            try:
                center_data = cluster_df[['x1', 'y1']].values
                hull = ConvexHull(center_data)
                self.hull.append(hull)
                sf_tmp += hull.area

                center_data = cluster_df[['x2', 'y1']].values
                # print(center_data)
                hull = ConvexHull(center_data)
                self.hull.append(hull)
                sf_tmp += hull.area

                center_data = cluster_df[['x1', 'y2']].values
                hull = ConvexHull(center_data)
                self.hull.append(hull)
                sf_tmp += hull.area

                center_data = cluster_df[['x2', 'y2']].values
                hull = ConvexHull(center_data)
                self.hull.append(hull)
                sf_tmp += hull.area

                self.prediction_surface = sf_tmp
            except:
                self.prediction_surface = -1

        return self.prediction_surface

        # if len(boxes) <= 2:
        #     return self.prediction_surface
        #
        # def create_corner_points():
        #     corner_points = [[], [], [], []]
        #     # print(corner_points)
        #     for box in boxes:
        #         corner_points[0].append([box[0], box[1]])
        #         corner_points[1].append([box[2], box[3]])
        #         corner_points[2].append([box[0], box[3]])
        #         corner_points[3].append([box[2], box[1]])
        #     return corner_points
        #
        # c_points = create_corner_points()
        # # print(c_points)
        #
        # ps = 0
        # for c_point in c_points:
        #     # if len(c_point) > 2:
        #     hull = ConvexHull(c_point)
        #     self.prediction_surface += hull.area
        # return self.prediction_surface


if __name__ == '__main__':
    uq_metrics = UQMetrics()
    uq_metrics.calcu_entropy([1/3, 1/3, 1/3])
    print(uq_metrics.shannon_entropy)

    print(uq_metrics.cal_vr(events=[]))
    # bs = [
    #     [
    #         1013.3162231445312,
    #         1310.352294921875,
    #         1118.556884765625,
    #         1385.857177734375
    #     ],
    #     [
    #         1014.5834350585938,
    #         1308.5045166015625,
    #         1121.2974853515625,
    #         1388.34228515625
    #     ],
    #     [
    #         1015.1859130859375,
    #         1308.117431640625,
    #         1119.5179443359375,
    #         1386.121826171875
    #     ]
    # ]
    #
    # uq_metrics.calcu_prediction_surface(bs)
