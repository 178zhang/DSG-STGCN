import time
import scipy.io
import scipy.io as sio
import h5py
import glob
import os
import torch
import mne
import numpy as np

device = torch.device("cpu")


# Functions:
def minimum_norm_estimate_3(x, leadfield, sensorNoise, tikhonov=0.05):
    ''' Based on gramfort et al https://www.sciencedirect.com/science/article/pii/S1053811920309150'''
    K_mne = np.matmul(leadfield.T, np.linalg.inv(np.matmul(leadfield, leadfield.T) + tikhonov**2 * sensorNoise))
    y_est = np.matmul(K_mne, x)
    return y_est


# Main ========================================================================
dataset = "../dataset_regions_4_neighbors_3_SNR_Source_10000_SNR_Sensor_40.mat"
lead_field_set = "../Lead_Field.mat"
saveset = "../MNE_result_regions_4_neighbors_3_SNR_Source_10000_SNR_Sensor_40.mat"

region_set = [4]
neighbor_set = [3] #[3,2,1]
SNR_set = [40] #[40,30,20,10]

for regions in region_set:
    for neighbors in neighbor_set:
        for SNR in SNR_set:
            # Load test set: EEG & source
            data = h5py.File(dataset)
            source = data['source'][()]
            EEG = data['EEG'][()]
            # load lead_field
            data_field = h5py.File(lead_field_set)
            lead_field = data_field['L'][()]  # .T

            source = source.T
            EEG = EEG.T
            lead_field = lead_field.T
            print("读取测试集数据形状：")
            print(source.shape)  # (2052, 20520)
            print(EEG.shape)  # (128, 20520)
            print("读取 lead field 形状：")
            print(lead_field.shape)  # (128, 2052)

            Estimate_MNE = np.empty(shape=(source.shape[0], source.shape[1]))

            # 样本个数 20520
            index = source.shape[1]
            print("index:", index)
            for i in range(index):
                data_eeg = EEG[:, i]  # EEG
                data_source = source[:, i]  # SOURCE

                # MNE
                sensorNoise = np.identity(lead_field.shape[0])  # x * rms(x) * 0.5  # some sensor noise
                y_est_MNE = minimum_norm_estimate_3(data_eeg, lead_field, sensorNoise, tikhonov=0.05)  # 1.62
                Estimate_MNE[:, i] = y_est_MNE

            # ========= save test result: =========
            # 保存为 mat 文件：
            sio.savemat(saveset, {'s_pred': Estimate_MNE})
            print("result MNE saved in: " + saveset)








