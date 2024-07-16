#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.05.2024 11:38
# @Author  : Chengjie
# @File    : utils.py
# @Software: PyCharm
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN


# https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan
# https://www.kaggle.com/discussions/questions-and-answers/166388


class DBSCANCluster:
    def __init__(self, x, eps=8.5, min_samples=8):
        # self.cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        # try:
        self.cluster = HDBSCAN(min_cluster_size=3).fit(x)
        #except:
        #    print('except')
        #    self.cluster = HDBSCAN(min_cluster_size=2).fit(x)

        self.mc_locations = np.c_[x, self.cluster.labels_.ravel()]
        self.mc_locations_df = pd.DataFrame(data=self.mc_locations, columns=['x1', 'y1', 'x2', 'y2',
                                                                             'center_x', 'center_y', 'label'])
        self.cluster_labels = np.unique(self.mc_locations[:, len(self.mc_locations[0]) - 1])

    def cluster_preds(self, preds):
        pred_id = 0
        preds_new = {}

        for cluster_label in self.cluster_labels:
            cluster_df = self.mc_locations_df.query('label == ' + str(cluster_label))
            boxs = cluster_df[['x1', 'y1', 'x2', 'y2']].values
            t = 0
            for key in preds.keys():
                if preds[key]['box'] in boxs and t == 0:
                    preds_new.update({'label_{}'.format(pred_id):
                        {
                            'box': [preds[key]['box']],
                            'label': [preds[key]['label']],
                            'score': [preds[key]['score']],
                            'logit': [preds[key]['logit']],
                            # 'center_point': [preds[key]['center_point']]
                        }
                    })
                    t = 1
                elif preds[key]['box'] in boxs and t != 0:
                    preds_new['label_{}'.format(pred_id)]['box'].append(preds[key]['box'])
                    preds_new['label_{}'.format(pred_id)]['label'].append(preds[key]['label'])
                    preds_new['label_{}'.format(pred_id)]['score'].append(preds[key]['score'])
                    preds_new['label_{}'.format(pred_id)]['logit'].append(preds[key]['logit'])

            pred_id += 1

        return preds_new


def cluster(mc_locations):
    clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)
    mc_locations = np.c_[mc_locations, clustering.labels_.ravel()]

    mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1', 'y1', 'x2', 'y2', 'label'])

    cluster_labels = np.unique(mc_locations[:, len(mc_locations[0]) - 1])
    total_cluster_surface = 0.0
    avg_surface = -1.0
    for cluster_label in cluster_labels:
        sf_tmp = 0
        cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
        if cluster_df.shape[0] > 2:
            center_data = cluster_df[['x1', 'y1']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[['x2', 'y1']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[['x1', 'y2']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area

            center_data = cluster_df[['x2', 'y2']].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area
        # print(sf_tmp)
        avg_surface = total_cluster_surface / mc_locations.shape[0]

    # print(total_cluster_surface, avg_surface)


# def dbscan_cluster(mc_locations):


def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, k - 1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8, 8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()
