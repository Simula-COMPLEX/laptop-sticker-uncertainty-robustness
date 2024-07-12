import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, pearsonr, spearmanr
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.multitest import multipletests


class StatisticalTest:

    def __init__(self):
        self.model_list = ['fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
                           'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
                           'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large']
        self.dropout_list = ['0.0', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']
        self.dataset_list = ['origimg', 'dalleimg', 'sdimg']
        self.adv_list = ['org', 'adv_run_1', 'adv_run_2', 'adv_run_3', 'adv_run_4', 'adv_run_5',
                         'adv_run_6', 'adv_run_7', 'adv_run_8', 'adv_run_9', 'adv_run_10']
        self.metric_list = ['MAP', 'UQ[VR]', 'UQ[IE]', 'UQ[MI]', 'UQ[TR]', 'UQ[PS]']
        self.UQMs_list = ['UQ[VR]', 'UQ[IE]', 'UQ[MI]', 'UQ[TR]', 'UQ[PS]']

    def extract_number(self, image_name):
        match = re.search(r'\d+(?=\.[^.]*$)', image_name)
        if match:
            return int(match.group())
        else:
            return None

    def filter_experiment_results_org(self):
        for dropout in self.dropout_list:
            for dataset in self.dataset_list:
                model_df_list = []
                for model in self.model_list:
                    file_path = f"../experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                    df = pd.read_csv(file_path, skipfooter=1, engine='python')
                    if dropout != '0.0':
                        df = df[df['UQ[PS]'] != -1]
                    model_df_list.append(df)
                image_names = [set(df['image_name']) for df in model_df_list]
                common_image_names = set.intersection(*image_names)
                filtered_model_df_list = [df[df['image_name'].isin(common_image_names)] for df in model_df_list]
                common_image_names = sorted(common_image_names, key=self.extract_number)
                for i, df in enumerate(filtered_model_df_list):
                    filtered_model_df_list[i] = df.set_index('image_name').loc[common_image_names].reset_index()
                for i, df in enumerate(filtered_model_df_list):
                    file_path = f"./experiment_results/{self.model_list[i]}"
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    df.to_csv(f"{file_path}/logs_{dropout}_{dataset}-org.csv", index=False)

    def filter_experiment_results_org_check(self):
        for dropout in self.dropout_list:
            for dataset in self.dataset_list:
                model_df_list = []
                for model in self.model_list:
                    file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                    df = pd.read_csv(file_path, skipfooter=1, engine='python')
                    if dropout != '0.0':
                        df = df[df['UQ[PS]'] != -1]
                    model_df_list.append(df)
                image_names = [df['image_name'].reset_index(drop=True) for df in model_df_list]
                for i in range(1, len(image_names)):
                    if not image_names[0].equals(image_names[i]):
                        print(f"{dropout}_{dataset}_{self.model_list[i]}-org.csv")

    def filter_experiment_results_adv(self):
        for dropout in self.dropout_list:
            for dataset in self.dataset_list:
                model_df_list = []
                for model in self.model_list:
                    for adv in self.adv_list:
                        file_path = f"../experiment_results/{model}/logs_{dropout}_{dataset}-{adv}.csv"
                        df = pd.read_csv(file_path, skipfooter=1, engine='python')
                        if dropout != '0.0':
                            df = df[df['UQ[PS]'] != -1]
                        model_df_list.append(df)
                image_names = [set(df['image_name'].apply(lambda x: x.split('/')[-1])) for df in model_df_list]
                common_image_names = set.intersection(*image_names)
                sorted_common_image_numbers = sorted(common_image_names, key=self.extract_number)

                filtered_model_df_list = []
                for df in model_df_list:
                    df['split_image_name'] = df['image_name'].apply(lambda x: x.split('/')[-1])
                    filtered_df = df[df['split_image_name'].isin(common_image_names)]
                    filtered_df = filtered_df.set_index('split_image_name').loc[sorted_common_image_numbers].reset_index()
                    filtered_df = filtered_df.drop(columns=['split_image_name'])
                    filtered_df = filtered_df.set_index('image_name')
                    filtered_model_df_list.append(filtered_df)

                for i, df in enumerate(filtered_model_df_list):
                    file_path = f"./experiment_results_RS/{self.model_list[i // len(self.adv_list)]}"
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    adv_index = i % len(self.adv_list)
                    df.to_csv(f"{file_path}/logs_{dropout}_{dataset}-{self.adv_list[adv_index]}.csv", index=True)

    def filter_experiment_results_adv_check(self):
        for dropout in self.dropout_list:
            for dataset in self.dataset_list:
                model_df_list = []
                for model in self.model_list:
                    for adv in self.adv_list:
                        file_path = f"./experiment_results_RS/{model}/logs_{dropout}_{dataset}-{adv}.csv"
                        df = pd.read_csv(file_path, skipfooter=1, engine='python')
                        if dropout != '0.0':
                            df = df[df['UQ[PS]'] != -1]
                        model_df_list.append(df)
                image_names = [df['image_name'].apply(lambda x: x.split('/')[-1]).reset_index(drop=True) for df in model_df_list]
                for i in range(len(self.model_list)):
                    for j in range(len(self.adv_list)):
                        if not image_names[0].equals(image_names[i * len(self.adv_list) + j]):
                            print(f"{dropout}_{dataset}_{self.model_list[i]}-{self.adv_list[j]}.csv")

    def calculate_mAP_RS(self, dropout, dataset):
        RS_data = []
        for model in self.model_list:
            file_path_org = f"./experiment_results_RS/{model}/logs_{dropout}_{dataset}-org.csv"
            df_org = pd.read_csv(file_path_org)
            df_org_mAP = df_org[['image_name', 'MAP']]
            sum_list = df_org['MAP'].tolist()
            diff_list = [0 for _ in range(df_org_mAP.shape[0])]
            for adv in self.adv_list:
                if adv == 'org':
                    continue
                file_path_adv = f"./experiment_results_RS/{model}/logs_{dropout}_{dataset}-{adv}.csv"
                df_adv = pd.read_csv(file_path_adv)
                df_adv_mAP = df_adv[['image_name', 'MAP']]
                for index, row in df_org_mAP.iterrows():
                    key = row['image_name'].split('/')[-1]
                    mAP_org = row['MAP']
                    if key in df_adv_mAP['image_name'].apply(lambda x: x.split('/')[-1]).values:
                        mAP_adv = df_adv_mAP[df_adv_mAP['image_name'].apply(lambda x: x.split('/')[-1]) == key]['MAP'].iloc[0]
                        sum_list[index] += mAP_adv
                        diff_list[index] += abs(mAP_org - mAP_adv)
                    else:
                        print(f"{dropout}_{dataset}_{model}_{adv}_{key}")
            sum_list = [x / 11 for x in sum_list]
            diff_list = [x / 10 for x in diff_list]
            RS_list = [x - y for x, y in zip(sum_list, diff_list)]
            RS_data.append(RS_list)
        return RS_data

    def calculate_UQMs_RS(self, dropout, dataset):
        RS_data = []
        columns = ['image_name'] + self.UQMs_list
        for model in self.model_list:
            file_path_org = f"./experiment_results_RS/{model}/logs_{dropout}_{dataset}-org.csv"
            df_org = pd.read_csv(file_path_org)
            df_org_UQMs = df_org[columns]
            UQMs_sum_dic = {x: (df_org[x] / (df_org[x] + 1)).tolist() for x in self.UQMs_list}
            UQMs_diff_dic = {x: [0 for _ in range(df_org_UQMs.shape[0])] for x in self.UQMs_list}
            for adv in self.adv_list:
                if adv == 'org':
                    continue
                file_path_adv = f"./experiment_results_RS/{model}/logs_{dropout}_{dataset}-{adv}.csv"
                df_adv = pd.read_csv(file_path_adv)
                df_adv_UQMs = df_adv[columns]
                for index, row in df_org_UQMs.iterrows():
                    key = row['image_name'].split('/')[-1]
                    if key in df_adv_UQMs['image_name'].apply(lambda x: x.split('/')[-1]).values:
                        for UQM in self.UQMs_list:
                            UQM_adv = df_adv_UQMs[df_adv_UQMs['image_name'].apply(lambda x: x.split('/')[-1]) == key][UQM].iloc[0]
                            UQM_adv = UQM_adv / (UQM_adv + 1)
                            UQMs_sum_dic[UQM][index] += UQM_adv
                            UQMs_diff_dic[UQM][index] += abs(row[UQM] / (row[UQM] + 1) - UQM_adv)
                    else:
                        print(f"{dropout}_{dataset}_{model}_{adv}_{key}")
            RS_list = {}
            for UQM in self.UQMs_list:
                UQMs_sum_dic[UQM] = [x / 11 for x in UQMs_sum_dic[UQM]]
                UQMs_diff_dic[UQM] = [x / 10 for x in UQMs_diff_dic[UQM]]
                RS_list[UQM] = [1 - (x + y) for x, y in zip(UQMs_sum_dic[UQM], UQMs_diff_dic[UQM])]
            RS_UQMs_list = [0 for _ in range(len(RS_list[self.UQMs_list[0]]))]
            for UQM in self.UQMs_list:
                RS_UQMs_list = [x + y for x, y in zip(RS_UQMs_list, RS_list[UQM])]
            RS_UQMs_list = [x / len(self.UQMs_list) for x in RS_UQMs_list]
            RS_data.append(RS_UQMs_list)
        return RS_data

    def mAP_UQMs_RS_save(self):
        for dropout in self.dropout_list:
            if dropout == '0.0':
                continue
            for dataset in self.dataset_list:
                file_path = f"./experiment_results_RS/{self.model_list[0]}/logs_{dropout}_{dataset}-org.csv"
                df = pd.read_csv(file_path)
                df_image = df[['image_name', 'image']]

                RS_mAP = self.calculate_mAP_RS(dropout, dataset)
                RS_UQMs = self.calculate_UQMs_RS(dropout, dataset)

                for i, model in enumerate(self.model_list):
                    RS_mAP_column_df = pd.DataFrame(RS_mAP[i], columns=['RS'])
                    RS_mAP_df = pd.concat([df_image, RS_mAP_column_df], axis=1)
                    RS_mAP_folder_path = f'./RS/mAP/{model}'
                    if not os.path.exists(RS_mAP_folder_path):
                        os.makedirs(RS_mAP_folder_path)
                    RS_mAP_file_path = os.path.join(RS_mAP_folder_path, f'logs_{dropout}_{dataset}-RS.csv')
                    RS_mAP_df.to_csv(RS_mAP_file_path, index=False)

                    RS_UQMs_column_df = pd.DataFrame(RS_UQMs[i], columns=['RS'])
                    RS_UQMs_df = pd.concat([df_image, RS_UQMs_column_df], axis=1)
                    RS_UQMs_folder_path = f'./RS/UQMs/{model}'
                    if not os.path.exists(RS_UQMs_folder_path):
                        os.makedirs(RS_UQMs_folder_path)
                    RS_UQMs_file_path = os.path.join(RS_UQMs_folder_path, f'logs_{dropout}_{dataset}-RS.csv')
                    RS_UQMs_df.to_csv(RS_UQMs_file_path, index=False)

    def mAP_UQMs_Friedman_Wilcoxon_Holm_R_Mean(self):
        friedman_folder_path = './friedman'
        if not os.path.exists(friedman_folder_path):
            os.makedirs(friedman_folder_path)
        friedman_file_path = os.path.join(friedman_folder_path, 'friedman_results.csv')
        whrm_folder_path = './whrm'
        if not os.path.exists(whrm_folder_path):
            os.makedirs(whrm_folder_path)

        for dropout in self.dropout_list:
            if dropout == '0.0':
                continue
            for dataset in self.dataset_list:
                friedman_data = {'Dropout': [dropout], 'Dataset': [dataset]}
                friedman_df = pd.DataFrame(friedman_data)
                org_RS_list = self.metric_list + ['mAP_RS', 'UQMs_RS']
                for metric in org_RS_list:
                    whrm_file_path = os.path.join(whrm_folder_path, f'{metric}_results.csv')

                    model_metric_list = []
                    if metric in self.metric_list:
                        for model in self.model_list:
                            file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                            df = pd.read_csv(file_path)
                            model_metric_list.append(df[metric])
                    else:
                        if metric == 'mAP_RS':
                            for model in self.model_list:
                                file_path = f"./RS/mAP/{model}/logs_{dropout}_{dataset}-RS.csv"
                                df = pd.read_csv(file_path)
                                model_metric_list.append(df['RS'])
                        else:
                            for model in self.model_list:
                                file_path = f"./RS/UQMs/{model}/logs_{dropout}_{dataset}-RS.csv"
                                df = pd.read_csv(file_path)
                                model_metric_list.append(df['RS'])

                    friedman_statistic, friedman_p = friedmanchisquare(*model_metric_list)
                    friedman_df[f'{metric}_Friedman_p_value'] = friedman_p
                    if friedman_p < 0.01:
                        p_value_sign = '< 0.01'
                    elif friedman_p < 0.05:
                        p_value_sign = '< 0.05'
                    else:
                        p_value_sign = '>= 0.05'
                    friedman_df[f'{metric}_Friedman_p'] = p_value_sign
                    friedman_df[f'{metric}_Friedman_statistic'] = friedman_statistic

                    model_com_name_list = []
                    p_value_list = []
                    W_pos_list = []
                    W_neg_list = []
                    rbc_list = []
                    r_list = []
                    rb_list = []
                    diff_mean_list = []
                    for i in range(len(model_metric_list) - 1):
                        for j in range(i + 1, len(model_metric_list)):
                            model_com_name_list.append(f"{self.model_list[i]} {self.model_list[j]}")
                            # check the pairwise difference
                            diff = model_metric_list[i] - model_metric_list[j]
                            # plt.hist(diff, bins=10, edgecolor='black')
                            # plt.xlabel('Difference')
                            # plt.ylabel('Frequency')
                            # plt.title('Distribution of Paired Differences')
                            # plt.show()
                            diff = np.array(diff.tolist())
                            median_diff = np.median(diff)
                            mad_diff = np.median(np.abs(diff - median_diff))
                            lower_bound = median_diff - 2 * mad_diff
                            upper_bound = median_diff + 2 * mad_diff
                            if lower_bound <= 0 <= upper_bound:
                                alternative = 'two-sided'
                            elif median_diff > 0:
                                alternative = 'greater'
                            elif median_diff < 0:
                                alternative = 'less'
                            else:
                                alternative = 'two-sided'
                            if np.all(diff == 0):
                                zero_method = 'zsplit'
                            else:
                                zero_method = 'pratt'
                            # alternative='two-sided' 'greater' 'less'
                            statistic, p_value = wilcoxon(model_metric_list[i].tolist(), model_metric_list[j].tolist(),
                                                          zero_method=zero_method, alternative=alternative, nan_policy='raise')
                            p_value_list.append(p_value)
                            # rank-biserial correlation
                            # correlation coefficient r
                            # matched-pairs rank-biserial correlation coefficient
                            diff = model_metric_list[i] - model_metric_list[j]
                            non_zero_diff = diff[diff != 0]
                            abs_diff = np.abs(non_zero_diff)
                            ranks = rankdata(abs_diff)
                            W_pos = np.sum(ranks[non_zero_diff > 0])
                            W_neg = np.sum(ranks[non_zero_diff < 0])
                            T = np.min([W_pos, W_neg])
                            W_pos_list.append(W_pos)
                            W_neg_list.append(W_neg)
                            n = len(non_zero_diff)
                            rbc = (W_pos - W_neg) / (n * (n + 1) / 2)
                            rbc_list.append(rbc)
                            z = (T - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                            r = z / np.sqrt(n)
                            r_list.append(r)
                            rb = (4 * np.abs(T - ((W_pos + W_neg) / 2))) / (n * (n + 1))
                            rb_list.append(rb)
                            # mean distance
                            non_zero_diff = non_zero_diff / (non_zero_diff + 1)
                            diff_mean_list.append(non_zero_diff.mean())
                    # Holm–Bonferroni method
                    reject, corrected_p_values, _, _ = multipletests(p_value_list, alpha=0.05, method='holm')
                    for i, name in enumerate(model_com_name_list):
                        if corrected_p_values[i] < 0.01:
                            p_values_sign = '< 0.01'
                        elif corrected_p_values[i] < 0.05:
                            p_values_sign = '< 0.05'
                        else:
                            p_values_sign = '>= 0.05'
                        if 0.0 < abs(rbc_list[i]) < 0.1:
                            rbc_sign = 'negligible'
                        elif 0.1 <= abs(rbc_list[i]) < 0.3:
                            rbc_sign = 'small'
                        elif 0.3 <= abs(rbc_list[i]) < 0.5:
                            rbc_sign = 'medium'
                        elif 0.5 <= abs(rbc_list[i]) <= 1.0:
                            rbc_sign = 'large'
                        else:
                            rbc_sign = 'no'
                        if 0.0 < abs(r_list[i]) < 0.1:
                            r_sign = 'negligible'
                        elif 0.1 <= abs(r_list[i]) < 0.3:
                            r_sign = 'small'
                        elif 0.3 <= abs(r_list[i]) < 0.5:
                            r_sign = 'medium'
                        elif 0.5 <= abs(r_list[i]) <= 1.0:
                            r_sign = 'large'
                        else:
                            r_sign = 'no'
                        if 0.0 < abs(rb_list[i]) < 0.1:
                            rb_sign = 'negligible'
                        elif 0.1 <= abs(rb_list[i]) < 0.3:
                            rb_sign = 'small'
                        elif 0.3 <= abs(rb_list[i]) < 0.5:
                            rb_sign = 'medium'
                        elif 0.5 <= abs(rb_list[i]) <= 1.0:
                            rb_sign = 'large'
                        else:
                            rb_sign = 'no'
                        if metric == 'MAP' or metric == 'mAP_RS' or metric == 'UQMs_RS':
                            if rbc_list[i] > 0:
                                b_rbc_sign = name.split(' ')[0]
                            elif rbc_list[i] < 0:
                                b_rbc_sign = name.split(' ')[-1]
                            else:
                                b_rbc_sign = 'NO'
                            if diff_mean_list[i] > 0:
                                b_m_sign = name.split(' ')[0]
                            elif diff_mean_list[i] < 0:
                                b_m_sign = name.split(' ')[-1]
                            else:
                                b_m_sign = 'NO'
                        else:
                            if rbc_list[i] > 0:
                                b_rbc_sign = name.split(' ')[-1]
                            elif rbc_list[i] < 0:
                                b_rbc_sign = name.split(' ')[0]
                            else:
                                b_rbc_sign = 'NO'
                            if diff_mean_list[i] > 0:
                                b_m_sign = name.split(' ')[-1]
                            elif diff_mean_list[i] < 0:
                                b_m_sign = name.split(' ')[0]
                            else:
                                b_m_sign = 'NO'
                        whrm_data = {'Dropout': [dropout], 'Dataset': [dataset], 'Model_1': [name.split(' ')[0]],
                                     'Model_2': [name.split(' ')[-1]], 'p_value': [corrected_p_values[i]],
                                     'p': [p_values_sign], 'MD_value': [diff_mean_list[i]], 'Better_MD': [b_m_sign],
                                     'rbc_value': [rbc_list[i]], 'rbc': [rbc_sign], 'Better_rbc': [b_rbc_sign],
                                     'r_value': [abs(r_list[i])], 'r': [r_sign],
                                     'rb_value': [abs(rb_list[i])], 'rb': [rb_sign]}
                        whrm_df = pd.DataFrame(whrm_data)
                        whrm_df.to_csv(whrm_file_path, mode='a', header=not os.path.exists(whrm_file_path), index=False)
                friedman_df.to_csv(friedman_file_path, mode='a', header=not os.path.exists(friedman_file_path), index=False)

    def mAP_UQMs_Rank_Condifence_Order(self):
        whrm_folder_path = './whrm'
        org_RS_list = self.metric_list + ['mAP_RS', 'UQMs_RS']
        for metric in org_RS_list:
            rco_file_path = os.path.join(whrm_folder_path, f'{metric}_rco_results.csv')
            whrm_file_path = os.path.join(whrm_folder_path, f'{metric}_results.csv')
            whrm_df = pd.read_csv(whrm_file_path)
            for dropout in self.dropout_list:
                if dropout == '0.0':
                    continue
                for dataset in self.dataset_list:
                    filtered_df = whrm_df[(whrm_df['Dropout'] == float(dropout)) & (whrm_df['Dataset'] == dataset)]
                    model_order = list(self.model_list)
                    for i in range(len(model_order) - 1):
                        for j in range(i + 1, len(model_order)):
                            condition = (filtered_df['Model_1'] == model_order[i]) & \
                                        (filtered_df['Model_2'] == model_order[j])
                            if condition.any():
                                model_filtered_df = filtered_df[condition]
                            else:
                                model_filtered_df = filtered_df[(filtered_df['Model_2'] == model_order[i]) &
                                                                (filtered_df['Model_1'] == model_order[j])]
                            if model_filtered_df['p_value'].values[0] < 0.05 and \
                                (model_filtered_df['rbc'].values[0] == 'medium' or
                                 model_filtered_df['rbc'].values[0] == 'large'):
                                if model_filtered_df['Better_rbc'].values[0] == model_order[i]:
                                    model_ = model_order[i]
                                    model_order[i] = model_order[j]
                                    model_order[j] = model_
                    model_rank = [1 for _ in range(len(model_order))]
                    for i in range(1, len(model_order)):
                        condition = (filtered_df['Model_1'] == model_order[i]) & \
                                    (filtered_df['Model_2'] == model_order[i-1])
                        if condition.any():
                            model_filtered_df = filtered_df[condition]
                        else:
                            model_filtered_df = filtered_df[(filtered_df['Model_2'] == model_order[i]) &
                                                            (filtered_df['Model_1'] == model_order[i-1])]
                        if model_filtered_df['p_value'].values[0] < 0.05 and \
                                (model_filtered_df['rbc'].values[0] == 'medium' or
                                 model_filtered_df['rbc'].values[0] == 'large'):
                            if model_filtered_df['Better_rbc'].values[0] == model_order[i]:
                                model_rank[i] = model_rank[i - 1] + 1
                            else:
                                model_rank[i] = model_rank[i - 1]
                        else:
                            model_rank[i] = model_rank[i - 1]
                    model_confidence = [x / sum(model_rank) for x in model_rank]

                    models_with_ranks = list(zip(model_order, model_rank))
                    model_order_index = {model: index for index, model in enumerate(self.model_list)}

                    def rank_model_sort(item):
                        rank = item[1]
                        order_index = model_order_index[item[0]]
                        return (-rank, order_index)

                    models_with_ranks.sort(key=rank_model_sort)
                    better_result = ""
                    previous_rank = None
                    for model, rank in models_with_ranks:
                        if previous_rank is None or rank != previous_rank:
                            if previous_rank is not None:
                                better_result += ">"
                            better_result += model
                        else:
                            better_result += "=" + model
                        previous_rank = rank

                    def sort_key(model_tuple):
                        return self.model_list.index(model_tuple[0])

                    models_with_ranks = list(zip(model_order, model_rank, model_confidence))
                    sorted_models_with_ranks = sorted(models_with_ranks, key=sort_key)
                    rco_data = {'Dropout': [dropout], 'Dataset': [dataset], 'Better': [better_result]}
                    for model, rank, confidence in sorted_models_with_ranks:
                        rco_data[f"{model}_rank"] = rank
                        rco_data[f"{model}_confidence"] = confidence
                    rco_df = pd.DataFrame(rco_data)
                    rco_df.to_csv(rco_file_path, mode='a', header=not os.path.exists(rco_file_path), index=False)

    def mAP_UQMs_Order_Combine(self):
        whrm_folder_path = './whrm'
        org_RS_list = self.metric_list + ['mAP_RS', 'UQMs_RS']
        order_combine_file = os.path.join(whrm_folder_path, f'order_results.csv')
        file_path = os.path.join(whrm_folder_path, f'{org_RS_list[0]}_rco_results.csv')
        combine_df = pd.read_csv(file_path, usecols=[0, 1, 2])
        columns_name = ['Dropout', 'Dataset', org_RS_list[0]]
        combine_df.columns = columns_name
        for index in range(1, len(org_RS_list)):
            file_path = os.path.join(whrm_folder_path, f'{org_RS_list[index]}_rco_results.csv')
            file = pd.read_csv(file_path, usecols=[0, 1, 2])
            combine_df = pd.merge(combine_df, file, on=['Dropout', 'Dataset'])
            columns_name.append(org_RS_list[index])
            combine_df.columns = columns_name
        combine_df.to_csv(order_combine_file, index=False)

        def replace_model(df, old_str, new_str):
            def replace(value):
                if isinstance(value, str):
                    return value.replace(old_str, new_str)
                return value

            return df.applymap(replace)

        model_list = ['fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_resnet50_fpn',
                      'retinanet_resnet50_fpn_v2', 'retinanet_resnet50_fpn',
                      'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large']
        model_name = ['M2', 'M1', 'M4', 'M3', 'M5', 'M6']
        for index, model in enumerate(model_list):
            combine_df = replace_model(combine_df, model, model_name[index])
        order_combine_file = os.path.join(whrm_folder_path, f'order_results_.csv')
        combine_df.to_csv(order_combine_file, index=False)

        def get_best(value):
            if isinstance(value, str):
                value = value.split('>')[0]
                value = value.replace('=', '/')
                return value
            return value

        combine_df = combine_df.applymap(get_best)
        order_combine_file = os.path.join(whrm_folder_path, f'order_results_best.csv')
        combine_df.to_csv(order_combine_file, index=False)

    def mAP_UQMs_Pearson_Spearman_org(self):
        correlation_folder_path = './correlation'
        if not os.path.exists(correlation_folder_path):
            os.makedirs(correlation_folder_path)
        correlation_file_path = os.path.join(correlation_folder_path, 'correlation_results.csv')
        for dropout in self.dropout_list:
            if dropout == '0.0':
                continue
            for dataset in self.dataset_list:
                for model in self.model_list:
                    file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                    df = pd.read_csv(file_path)
                    model_metric_list = []
                    for metric in self.metric_list:
                        model_metric_list.append(df[metric])
                    metric_com_name_list = []
                    pearson_p_value_list = []
                    pearson_statistic_list = []
                    spearman_p_value_list = []
                    spearman_statistic_list = []
                    for i in range(1, len(model_metric_list)):
                        metric_com_name_list.append(f"{self.metric_list[0]} {self.metric_list[i]}")
                        statistic, p_value = pearsonr(model_metric_list[0].tolist(), model_metric_list[i].tolist(), alternative='less')
                        pearson_p_value_list.append(p_value)
                        pearson_statistic_list.append(statistic)
                        statistic, p_value = spearmanr(model_metric_list[0].tolist(), model_metric_list[i].tolist(), alternative='less')
                        spearman_p_value_list.append(p_value)
                        spearman_statistic_list.append(statistic)
                    # Holm–Bonferroni method
                    pearson_reject, pearson_corrected_p_values, _, _ = multipletests(pearson_p_value_list, alpha=0.05, method='holm')
                    spearman_reject, spearman_corrected_p_values, _, _ = multipletests(spearman_p_value_list, alpha=0.05, method='holm')
                    for i, name in enumerate(metric_com_name_list):
                        if pearson_corrected_p_values[i] < 0.01:
                            pearson_p_values_sign = '< 0.01'
                        elif pearson_corrected_p_values[i] < 0.05:
                            pearson_p_values_sign = '< 0.05'
                        else:
                            pearson_p_values_sign = '>= 0.05'
                        if spearman_corrected_p_values[i] < 0.01:
                            spearman_p_values_sign = '< 0.01'
                        elif spearman_corrected_p_values[i] < 0.05:
                            spearman_p_values_sign = '< 0.05'
                        else:
                            spearman_p_values_sign = '>= 0.05'
                        if 0.0 < abs(spearman_statistic_list[i]) < 0.3:
                            spearman_sign = 'negligible'
                        elif 0.3 <= abs(spearman_statistic_list[i]) < 0.5:
                            spearman_sign = 'low'
                        elif 0.5 <= abs(spearman_statistic_list[i]) < 0.7:
                            spearman_sign = 'moderate'
                        elif 0.7 <= abs(spearman_statistic_list[i]) < 0.9:
                            spearman_sign = 'high'
                        elif 0.9 <= abs(spearman_statistic_list[i]) <= 1.0:
                            spearman_sign = 'very high'
                        else:
                            spearman_sign = 'no'
                        correlation_data = {'Dropout': [dropout], 'Dataset': [dataset], 'Model': [model],
                                            'Metric_1': [name.split(' ')[0]], 'Metric_2': [name.split(' ')[-1]],
                                            'spearman_p_value': [spearman_corrected_p_values[i]], 'spearman_p': [spearman_p_values_sign],
                                            'spearman_statistic_value': [spearman_statistic_list[i]], 'spearman_statistic': [spearman_sign],
                                            'pearson_p_value': [pearson_corrected_p_values[i]], 'pearson_p': [pearson_p_values_sign],
                                            'pearson_statistic': [pearson_statistic_list[i]]}
                        correlation_df = pd.DataFrame(correlation_data)
                        correlation_df.to_csv(correlation_file_path, mode='a', header=not os.path.exists(correlation_file_path), index=False)

    def mAP_UQMs_Spearman_combine_org(self):
        correlation_folder_path = './correlation'
        if not os.path.exists(correlation_folder_path):
            os.makedirs(correlation_folder_path)
        for model in self.model_list:
            correlation_file_path = os.path.join(correlation_folder_path, f'correlation_results_dropout.csv')
            metrics_list = [[] for _ in range(len(self.metric_list))]
            metric_com_name_list = []
            pearson_p_value_list = []
            pearson_statistic_list = []
            spearman_p_value_list = []
            spearman_statistic_list = []
            for dropout in self.dropout_list:
                if dropout == '0.0':
                    continue
                for dataset in self.dataset_list:
                    file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                    df = pd.read_csv(file_path)
                    for i, metric in enumerate(self.metric_list):
                        metrics_list[i] += df[metric].tolist()
            for i in range(1, len(metrics_list)):
                metric_com_name_list.append(f"{self.metric_list[0]} {self.metric_list[i]}")
                statistic, p_value = pearsonr(metrics_list[0], metrics_list[i],
                                              alternative='less')
                pearson_p_value_list.append(p_value)
                pearson_statistic_list.append(statistic)
                statistic, p_value = spearmanr(metrics_list[0], metrics_list[i],
                                               alternative='less')
                spearman_p_value_list.append(p_value)
                spearman_statistic_list.append(statistic)
            # Holm–Bonferroni method
            pearson_reject, pearson_corrected_p_values, _, _ = multipletests(pearson_p_value_list, alpha=0.05,
                                                                             method='holm')
            spearman_reject, spearman_corrected_p_values, _, _ = multipletests(spearman_p_value_list, alpha=0.05,
                                                                               method='holm')
            for i, name in enumerate(metric_com_name_list):
                if pearson_corrected_p_values[i] < 0.01:
                    pearson_p_values_sign = '< 0.01'
                elif pearson_corrected_p_values[i] < 0.05:
                    pearson_p_values_sign = '< 0.05'
                else:
                    pearson_p_values_sign = '>= 0.05'
                if spearman_corrected_p_values[i] < 0.01:
                    spearman_p_values_sign = '< 0.01'
                elif spearman_corrected_p_values[i] < 0.05:
                    spearman_p_values_sign = '< 0.05'
                else:
                    spearman_p_values_sign = '>= 0.05'
                if 0.0 < abs(spearman_statistic_list[i]) < 0.3:
                    spearman_sign = 'negligible'
                elif 0.3 <= abs(spearman_statistic_list[i]) < 0.5:
                    spearman_sign = 'low'
                elif 0.5 <= abs(spearman_statistic_list[i]) < 0.7:
                    spearman_sign = 'moderate'
                elif 0.7 <= abs(spearman_statistic_list[i]) < 0.9:
                    spearman_sign = 'high'
                elif 0.9 <= abs(spearman_statistic_list[i]) <= 1.0:
                    spearman_sign = 'very high'
                else:
                    spearman_sign = 'no'
                correlation_data = {'Model': [model],
                                    'Metric_1': [name.split(' ')[0]],
                                    'Metric_2': [name.split(' ')[-1]],
                                    'spearman_p_value': [spearman_corrected_p_values[i]],
                                    'spearman_p': [spearman_p_values_sign],
                                    'spearman_statistic_value': [spearman_statistic_list[i]],
                                    'spearman_statistic': [spearman_sign],
                                    'pearson_p_value': [pearson_corrected_p_values[i]],
                                    'pearson_p': [pearson_p_values_sign],
                                    'pearson_statistic_value': [pearson_statistic_list[i]]}
                correlation_df = pd.DataFrame(correlation_data)
                correlation_df.to_csv(correlation_file_path, mode='a', header=not os.path.exists(correlation_file_path),
                                      index=False)

    def mAP_UQMs_Spearman_sort(self):
        correlation_folder_path = './correlation'
        correlation_file_path = os.path.join(correlation_folder_path, 'correlation_results.csv')
        df = pd.read_csv(correlation_file_path)
        sort_file_path = os.path.join(correlation_folder_path, 'correlation_results_.csv')
        for dropout in self.dropout_list:
            if dropout == '0.0':
                continue
            for dataset in self.dataset_list:
                for model in self.model_list:
                    sort_data = {'Dropout': [dropout], 'Dataset': [dataset], 'Model': [model]}
                    sort_df = pd.DataFrame(sort_data)
                    for metric in self.UQMs_list:
                        filtered_df = df[(df['Dropout'] == float(dropout)) & (df['Dataset'] == dataset) &
                                         (df['Model'] == model) & (df['Metric_2'] == metric)]
                        if pd.isna(filtered_df['spearman_p_value'].values[0]) or \
                                filtered_df['spearman_p_value'].values[0] == '':
                            sort_df[f'{metric}_p_value'] = 'NaN'
                            sort_df[f'{metric}_statistic'] = 'NaN'
                        else:
                            sort_df[f'{metric}_p_value'] = filtered_df['spearman_p'].values[0]
                            sort_df[f'{metric}_statistic'] = round(float(filtered_df['spearman_statistic_value'].values[0]), 3)
                    sort_df.to_csv(sort_file_path, mode='a', header=not os.path.exists(sort_file_path), index=False)

    def mAP_UQMs_Spearman_combine_sort(self):
        correlation_folder_path = './correlation'
        spearman_file_path = os.path.join(correlation_folder_path, f'correlation_results_dropout_spearman.csv')
        # pearson_file_path = os.path.join(correlation_folder_path, f'correlation_results_dropout_pearson.csv')
        correlation_file_path = os.path.join(correlation_folder_path, f'correlation_results_dropout.csv')
        df = pd.read_csv(correlation_file_path)
        for model in self.model_list:
            spearman_sort_data = {'Model': [model]}
            spearman_sort_df = pd.DataFrame(spearman_sort_data)
            # pearson_sort_data = {'Model': [model]}
            # pearson_sort_df = pd.DataFrame(pearson_sort_data)
            for metric in self.UQMs_list:
                filtered_df = df[(df['Model'] == model) & (df['Metric_2'] == metric)]
                if pd.isna(filtered_df['spearman_p_value'].values[0]) or \
                        filtered_df['spearman_p_value'].values[0] == '':
                    spearman_sort_df[f'{metric}_p_value'] = ['NaN']
                    spearman_sort_df[f'{metric}_statistic'] = ['NaN']
                else:
                    spearman_sort_df[f'{metric}_p_value'] = [filtered_df['spearman_p'].values[0]]
                    spearman_sort_df[f'{metric}_statistic'] = [round(float(filtered_df['spearman_statistic_value'].values[0]), 3)]
                # if pd.isna(filtered_df['pearson_p_value'].values[0]) or \
                #         filtered_df['pearson_p_value'].values[0] == '':
                #     pearson_sort_df[f'{metric}_p_value'] = ['NaN']
                #     pearson_sort_df[f'{metric}_statistic'] = ['NaN']
                # else:
                #     pearson_sort_df[f'{metric}_p_value'] = [filtered_df['pearson_p'].values[0]]
                #     pearson_sort_df[f'{metric}_statistic'] = [round(float(filtered_df['pearson_statistic_value'].values[0]), 3)]
            spearman_sort_df.to_csv(spearman_file_path, mode='a', header=not os.path.exists(spearman_file_path), index=False)
            # pearson_sort_df.to_csv(pearson_file_path, mode='a', header=not os.path.exists(pearson_file_path), index=False)

    def mAP_Dropout_plt(self):
        folder_path = './figure'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_list = []
        for metric in ['MAP']:
            data = {'Dropout': [], 'Value': [], 'Model': [], 'Dataset': [], 'Metric': []}
            for dataset in self.dataset_list:
                for i, model in enumerate(self.model_list):
                    for dropout in self.dropout_list:
                        file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                        df = pd.read_csv(file_path)
                        data['Dropout'].append(float(dropout))
                        data['Value'].append(df[metric].mean())
                        data['Model'].append(f"M{i + 1}")
                        data['Dataset'].append(dataset)
                        data['Metric'].append(metric)
            data_list.append(pd.DataFrame(data))
        data = pd.concat(data_list, ignore_index=True)

        num_datasets = len(self.dataset_list)
        num_metrics = len(['MAP'])
        fig, axes = plt.subplots(num_metrics, num_datasets, figsize=(5 * num_datasets, 4 * num_metrics), sharex=True)

        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (238, 155, 0), (9, 147, 150), (34, 96, 115)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]

        marker_list = ['o', 's', '^', 'v', 'D', 'p']
        markersize_list = [11, 10, 13, 13, 10, 13]

        dataset_name_list = ['origImg', 'dalleImg', 'stableImg']

        middle_col_idx = num_datasets // 2

        for i, metric in enumerate(['MAP']):
            y_min = float('inf')
            y_max = float('-inf')

            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                subset = data[(data['Metric'] == metric) & (data['Dataset'] == dataset)]
                sns.lineplot(data=subset, x='Dropout', y='Value', hue='Model', palette=sns.color_palette(color_list),
                             ax=ax)

                for k, model in enumerate(self.model_list):
                    model_data = subset[subset['Model'] == f'M{k + 1}']
                    x = model_data['Dropout']
                    y = model_data['Value']
                    ci = 1.96 * np.std(y) / np.sqrt(len(x))
                    ax.fill_between(x, y - ci, y + ci, color=color_list[k], alpha=0.2)

                    y_min = min(y_min, (y - ci).min())
                    y_max = max(y_max, (y + ci).max())

                lines = ax.get_lines()
                for line, marker, markersize in zip(lines, marker_list * len(self.model_list),
                                                    markersize_list * len(self.model_list)):
                    line.set_marker(marker)
                    line.set_markersize(markersize)

                ax.set_facecolor('#FAFAFA')
                ax.grid(True, which='both', linestyle='-', linewidth=0.5, color=to_rgb((210 / 255, 210 / 255, 210 / 255)))
                ax.tick_params(axis='both', labelsize=20)

                if i == 0:
                    ax.set_title(f'{dataset_name_list[j]}', fontweight='normal', fontstyle='italic', size=27, pad=12)
                else:
                    ax.set_title('')
                if i == num_metrics - 1 and j == middle_col_idx:
                    ax.set_xlabel('Dropout Rate', fontsize=25)
                    ax.set_xlim(-0.02, 0.52)
                    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
                    ax.set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=20)
                else:
                    ax.set_xlabel('')
                if j % len(self.dataset_list) == 0:
                    ax.set_ylabel(f'mAP', fontsize=25, fontweight='normal', fontstyle='italic')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)
                    spine.set_visible(True)

            y_gap = y_max * 0.05
            y_min -= y_gap
            y_max += y_gap
            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        handles, labels = axes[0, 0].get_legend_handles_labels() if num_metrics > 1 else axes[
            0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.legend().set_visible(False)
        fig.legend(handles, labels, fontsize=22, ncol=len(self.model_list), bbox_to_anchor=(0.512, -0.05),
                   loc='upper center', edgecolor=to_rgb((180 / 255, 180 / 255, 180 / 255)),
                   frameon=True, framealpha=1.0, facecolor='white')

        fig.subplots_adjust(hspace=0.1, wspace=0.05)

        file_path = os.path.join(folder_path, f'mAP.pdf')
        plt.savefig(file_path, bbox_inches='tight')

    def UQMs_Dropout_plt(self):
        folder_path = './figure'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_list = []
        for metric in self.UQMs_list:
            data = {'Dropout': [], 'Value': [], 'Model': [], 'Dataset': [], 'Metric': []}
            for dataset in self.dataset_list:
                for i, model in enumerate(self.model_list):
                    for dropout in self.dropout_list:
                        if dropout == '0.0':
                            continue
                        file_path = f"./experiment_results/{model}/logs_{dropout}_{dataset}-org.csv"
                        df = pd.read_csv(file_path)
                        data['Dropout'].append(float(dropout))
                        data['Value'].append(df[metric].mean())
                        data['Model'].append(f"M{i + 1}")
                        data['Dataset'].append(dataset)
                        data['Metric'].append(metric)
            data_list.append(pd.DataFrame(data))
        data = pd.concat(data_list, ignore_index=True)

        num_datasets = len(self.dataset_list)
        num_metrics = len(self.UQMs_list)
        fig, axes = plt.subplots(num_metrics, num_datasets, figsize=(5 * num_datasets, 4 * num_metrics), sharex=True)

        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (238, 155, 0), (9, 147, 150), (34, 96, 115)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]

        marker_list = ['o', 's', '^', 'v', 'D', 'p']
        markersize_list = [11, 10, 13, 13, 10, 13]

        dataset_name_list = ['origImg', 'dalleImg', 'stableImg']
        metric_name_list = ['VR', 'SE', 'MI', 'TV', 'PS']

        middle_col_idx = num_datasets // 2

        class MyScalarFormatter(ScalarFormatter):
            def _set_format(self):
                self.format = '%1.1f'

        for i, metric in enumerate(self.UQMs_list):
            y_min = float('inf')
            y_max = float('-inf')

            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                subset = data[(data['Metric'] == metric) & (data['Dataset'] == dataset)]
                sns.lineplot(data=subset, x='Dropout', y='Value', hue='Model', palette=sns.color_palette(color_list),
                             ax=ax)

                for k, model in enumerate(self.model_list):
                    model_data = subset[subset['Model'] == f'M{k + 1}']
                    x = model_data['Dropout']
                    y = model_data['Value']
                    ci = 1.96 * np.std(y) / np.sqrt(len(x))
                    ax.fill_between(x, y - ci, y + ci, color=color_list[k], alpha=0.2)

                    y_min = min(y_min, (y - ci).min())
                    y_max = max(y_max, (y + ci).max())

                lines = ax.get_lines()
                for line, marker, markersize in zip(lines, marker_list * len(self.model_list),
                                                    markersize_list * len(self.model_list)):
                    line.set_marker(marker)
                    line.set_markersize(markersize)

                ax.set_facecolor('#FAFAFA')
                ax.grid(True, which='both', linestyle='-', linewidth=0.5, color=to_rgb((210 / 255, 210 / 255, 210 / 255)))
                ax.tick_params(axis='both', labelsize=20)

                if i == 0:
                    ax.set_title(f'{dataset_name_list[j]}', fontweight='normal', fontstyle='italic', size=27, pad=12)
                else:
                    ax.set_title('')
                if i == num_metrics - 1 and j == middle_col_idx:
                    ax.set_xlabel('Dropout Rate', fontsize=25)
                    ax.set_xlim(0.08, 0.52)
                    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
                    ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=20)
                else:
                    ax.set_xlabel('')
                if j % len(self.dataset_list) == 0:
                    ax.set_ylabel(f'{metric_name_list[i]}', fontsize=25, fontweight='normal', fontstyle='italic')
                    formatter = MyScalarFormatter(useOffset=True, useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((0, 0))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.yaxis.get_offset_text().set_fontsize(20)
                    ax.yaxis.get_offset_text().set_color('black')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)
                    spine.set_visible(True)

            y_gap = y_max * 0.05
            y_min -= y_gap
            y_max += y_gap
            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        handles, labels = axes[0, 0].get_legend_handles_labels() if num_metrics > 1 else axes[
            0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.legend().set_visible(False)
        fig.legend(handles, labels, fontsize=22, ncol=len(self.model_list), bbox_to_anchor=(0.512, 0.078),
                   loc='upper center', edgecolor=to_rgb((180 / 255, 180 / 255, 180 / 255)),
                   frameon=True, framealpha=1.0, facecolor='white')

        fig.subplots_adjust(hspace=0.12, wspace=0.05)

        file_path = os.path.join(folder_path, f'UQMs.pdf')
        plt.savefig(file_path, bbox_inches='tight')

    def RS_Dropout_plt(self):
        folder_path = './figure'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_list = []
        for metric in ['mAP_RS', 'UQMs_RS']:
            data = {'Dropout': [], 'Value': [], 'Model': [], 'Dataset': [], 'Metric': []}
            for dataset in self.dataset_list:
                for i, model in enumerate(self.model_list):
                    for dropout in self.dropout_list:
                        if dropout == '0.0' and metric == 'UQMs_RS':
                            continue
                        if metric == 'mAP_RS':
                            file_path = f"./RS/mAP/{model}/logs_{dropout}_{dataset}-RS.csv"
                            df = pd.read_csv(file_path)
                        else:
                            file_path = f"./RS/UQMs/{model}/logs_{dropout}_{dataset}-RS.csv"
                            df = pd.read_csv(file_path)
                        data['Dropout'].append(float(dropout))
                        data['Value'].append(df['RS'].mean())
                        data['Model'].append(f"M{i + 1}")
                        data['Dataset'].append(dataset)
                        data['Metric'].append(metric)
            data_list.append(pd.DataFrame(data))
        data = pd.concat(data_list, ignore_index=True)

        num_datasets = len(self.dataset_list)
        num_metrics = len(['mAP_RS', 'UQMs_RS'])
        fig, axes = plt.subplots(num_metrics, num_datasets, figsize=(5 * num_datasets, 4 * num_metrics))

        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (238, 155, 0), (9, 147, 150), (34, 96, 115)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]

        marker_list = ['o', 's', '^', 'v', 'D', 'p']
        markersize_list = [11, 10, 13, 13, 10, 13]

        dataset_name_list = ['origImg', 'dalleImg', 'stableImg']

        middle_col_idx = num_datasets // 2

        class MyScalarFormatter(ScalarFormatter):
            def _set_format(self):
                self.format = '%1.1f'

        for i, metric in enumerate(['mAP_RS', 'UQMs_RS']):
            y_min = float('inf')
            y_max = float('-inf')

            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                subset = data[(data['Metric'] == metric) & (data['Dataset'] == dataset)]
                sns.lineplot(data=subset, x='Dropout', y='Value', hue='Model', palette=sns.color_palette(color_list),
                             ax=ax)

                for k, model in enumerate(self.model_list):
                    model_data = subset[subset['Model'] == f'M{k + 1}']
                    x = model_data['Dropout']
                    y = model_data['Value']
                    ci = 1.96 * np.std(y) / np.sqrt(len(x))
                    ax.fill_between(x, y - ci, y + ci, color=color_list[k], alpha=0.2)

                    y_min = min(y_min, (y - ci).min())
                    y_max = max(y_max, (y + ci).max())

                lines = ax.get_lines()
                for line, marker, markersize in zip(lines, marker_list * len(self.model_list),
                                                    markersize_list * len(self.model_list)):
                    line.set_marker(marker)
                    line.set_markersize(markersize)

                ax.set_facecolor('#FAFAFA')
                ax.grid(True, which='both', linestyle='-', linewidth=0.5, color=to_rgb((210 / 255, 210 / 255, 210 / 255)))
                ax.tick_params(axis='both', labelsize=20)

                if i == 0:
                    ax.set_title(f'{dataset_name_list[j]}', fontweight='normal', fontstyle='italic', size=27, pad=12)
                    ax.set_xlim(-0.02, 0.52)
                    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
                    ax.set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=20)
                else:
                    ax.set_title('')
                    ax.set_xlim(0.08, 0.52)
                    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
                    ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=20)
                if i == num_metrics - 1 and j == middle_col_idx:
                    ax.set_xlabel('Dropout Rate', fontsize=25)
                else:
                    ax.set_xlabel('')
                if j % len(self.dataset_list) == 0:
                    ax.set_ylabel('')
                    if i == 0:
                        ax.annotate('RS', xy=(-0.25, 0.35), xycoords='axes fraction', fontsize=25, fontweight='normal',
                                    fontstyle='italic', rotation=90)
                        ax.annotate('mAP', xy=(-0.2, 0.5), xycoords='axes fraction', fontsize=15, fontweight='normal',
                                    fontstyle='italic', rotation=90)
                    else:
                        ax.annotate('RS', xy=(-0.25, 0.4), xycoords='axes fraction', fontsize=25, fontweight='normal',
                                    fontstyle='italic', rotation=90)
                        ax.annotate('uq', xy=(-0.2, 0.55), xycoords='axes fraction', fontsize=15, fontweight='normal',
                                    fontstyle='italic', rotation=90)
                    formatter = MyScalarFormatter(useOffset=True, useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((0, 0))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.yaxis.get_offset_text().set_fontsize(20)
                    ax.yaxis.get_offset_text().set_color('black')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)
                    spine.set_visible(True)

            y_gap = y_max * 0.05
            y_min -= y_gap
            y_max += y_gap
            for j, dataset in enumerate(self.dataset_list):
                ax = axes[i, j] if num_metrics > 1 else axes[j]
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        handles, labels = axes[0, 0].get_legend_handles_labels() if num_metrics > 1 else axes[
            0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.legend().set_visible(False)
        fig.legend(handles, labels, fontsize=22, ncol=len(self.model_list), bbox_to_anchor=(0.512, 0.025),
                   loc='upper center', edgecolor=to_rgb((180 / 255, 180 / 255, 180 / 255)),
                   frameon=True, framealpha=1.0, facecolor='white')

        fig.subplots_adjust(hspace=0.23, wspace=0.05)

        file_path = os.path.join(folder_path, f'RS.pdf')
        plt.savefig(file_path, bbox_inches='tight')


if __name__ == '__main__':
    st = StatisticalTest()
    # org filter
    st.filter_experiment_results_org()
    st.filter_experiment_results_org_check()
    # adv RS filter
    st.filter_experiment_results_adv()
    st.filter_experiment_results_adv_check()
    # RS save
    st.mAP_UQMs_RS_save()
    # RQ1 RQ2
    st.mAP_UQMs_Friedman_Wilcoxon_Holm_R_Mean()
    st.mAP_UQMs_Rank_Condifence_Order()
    st.mAP_UQMs_Order_Combine()
    st.mAP_Dropout_plt()
    st.UQMs_Dropout_plt()
    st.RS_Dropout_plt()
    # RQ3
    st.mAP_UQMs_Pearson_Spearman_org()
    st.mAP_UQMs_Spearman_sort()
    st.mAP_UQMs_Spearman_combine_org()
    st.mAP_UQMs_Spearman_combine_sort()
