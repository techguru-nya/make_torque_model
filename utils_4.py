#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import glob
import shutil
import pandas as pd
import pickle
import numpy as np
import datetime
import zipfile
import matplotlib.pyplot as plt

import math
import matlab.engine

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import Common_3 as Common
import vRef_Evaluation_3 as vRef

import torch
from torch.nn import functional as F
# import torchvision
# from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset
import torch.nn.init as init
from sklearn import datasets
from sklearn.model_selection import train_test_split

# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# print(device)


# # Sim with CARSIM

# ## Convert ZIP(D97-->CSV)

# In[ ]:


# l_DATA = []
# l_DATA.append('c:\\TSDE_Workarea\\ktt2yk\\Projects\\22MY_3A0A_CR-V\\')
# PLT = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Tire_Torque\\Conv_Est_Tire.PLT'

# SAMPLING = 0.005 

# l_d_CSVs = []

# for DATA in l_DATA:
#     PATH_Search = [DATA]
#     PLT_Convert = PLT 
#     d_SIGNAL_PLT = Common.Select_Signal(PLT_Convert)
#     d_CSVs, l_Not_Convert = Common.SAVE_ZIP_to_CSV(d_SIGNAL_PLT, PATH_Search, SAMPLING, False)

#     PKL = f'{DATA}\\CSV_List.pkl'
#     with open(PKL, 'wb') as file:
#         pickle.dump(d_CSVs, file)
    
#     l_d_CSVs.append(d_CSVs)


# ## Data List

# In[4]:


# d_CSV = {}
# for d_ in l_d_CSVs:
#     for ZIP in d_:
#         Comment, CSV, _ = d_[ZIP]
#         ZIP = ZIP.replace('\\/', '\\')
#         d_CSV[CSV] = {'ZIP': ZIP, 'Comment': Comment}


# ## vRef

# In[5]:


# d_CSV_vRef = {}

# for i, CSV in enumerate(d_CSV):
#     updated_CSV = CSV
#     Out = vRef.vRef(CSV, updated_CSV, override=False)
#     if Out != None:      
#         d_CSV_vRef[CSV] = d_CSV[CSV]
#         print(f'{(i+1)}/{len(d_CSV)}, {CSV}: Success')
#     else:
#         print(f'{(i+1)}/{len(d_CSV)}, {CSV}: Failure')


# In[6]:


# PKL = f'Sim_vRef_List.pkl'
# with open(PKL, 'wb') as file:
#     pickle.dump(d_CSV_vRef, file)

# PKL = f'Sim_vRef_List.pkl'
# with open(PKL, 'rb') as file:
#     d_CSV_vRef= pickle.load(file)


# ## Selecting Data

# In[7]:


def Search_Comment(d_Datas, contain_list, not_contain_list, ignore_list=[]):
    Data_list = {}
    
    for File in d_Datas:
        s = str(os.path.basename(File)) + str(d_Datas[File]['ZIP']) + str(d_Datas[File]['Comment'])
        s = s.lower()

        for key in ignore_list:
            s = s.replace(key.lower(), '')
    
        contain = True
        if contain_list != []:
            for key in contain_list:
                if key.lower() not in s:
                    contain = False
              
        not_contain = True
        if  not_contain_list != []:
            for key in not_contain_list:
                if key.lower() in s:
                    not_contain = False
                    break

        if contain == True and not_contain == True:
            Data_list[File] = d_Datas[File]

    return Data_list


# In[8]:


def removing_duplication_files(Datas):
    file_list = {}
    not_duplication_files = []
    
    for label in Datas:
        # file = Datas[label]['ZIP']
        # root, ext = os.path.splitext(file)
        # zip_file = f'{root}.zip'
        zip_file = Datas[label]['ZIP']
        basename = os.path.basename(Datas[label]['ZIP'])
    
        if basename not in file_list.keys():
            datas = [zip_file]
            not_duplication_files.append(zip_file)
        else:
            datas = file_list[basename]
            
            for data in datas:
                zip_size = os.path.getsize(zip_file)
                data_size = os.path.getsize(data)
                if zip_size > data_size * 1.01 or zip_size < data_size * 0.99:
                    datas.append(zip_file)
                    not_duplication_files.append(zip_file)
            
        file_list[basename] = datas
    
    updated_Datas = {}
    for label in Datas:
        # file = Datas[label]['Features']
        # root, ext = os.path.splitext(file)
        # zip_file = f'{root}.zip'
        zip_file = Datas[label]['ZIP']
        if zip_file in not_duplication_files:
            updated_Datas[label] = Datas[label]

    return updated_Datas


# In[9]:


# データ選別
# contain_list = ['snow', '2wd']
# not_contain_list = ['split', 'abs', 'ice', 'hill', '10%', '4wd']
# ignore_list = []
# d_Selected_CSVs = Search_Comment(d_CSV_vRef, contain_list, not_contain_list, ignore_list=ignore_list)
# print(f'Snow: {len(d_Selected_CSVs)}')


# contain_list = ['ms']
# not_contain_list = ['summer', 'measure_v57_2wd', 'measure_v53_kh', 'measure_v75_2wd_wtsa', 'measure_v85_2wd_kyg']
# d_Selected_CSVs = Search_Comment(d_Selected_CSVs, contain_list, not_contain_list)
# print(f'MS: {len(d_Selected_CSVs)}')


# # 重複データの削除
# d_Selected_CSVs = removing_duplication_files(d_Selected_CSVs)
# print(f'Duplication: {len(d_Selected_CSVs)}')


# ## Estimate Wheel Torque

# In[10]:


def RunMatlab(Model, File, eng):
    _ = eng.sim(Model)
    
    while not os.path.exists(File):
        time.sleep(1)
        Time += 1
        
        if Time == 5:
            File = None
            break
    
    return File


# In[11]:


def MakeInputForEst(CSV, CSV_out):
    shutil.copy(CSV, CSV_out)
    
    df = pd.read_table(CSV_out, sep=",", index_col=None)

    for i in df.index:
        if i < 50:
            v = df.loc[i, 'vRef_sum_ax']
            # print(v)
            v_RL = df.loc[i, 'v_RR']
            v_RR = df.loc[i, 'v_RR']
            v_RA = (v_RL + v_RR) / 2
            if str(v) == 'nan':
                df.loc[i, 'vRef_sum_ax'] = v_RA
        else:
            break
    
    # 'seconds'列が存在し、秒単位のデータとして扱う
    df['timestamp'] = pd.to_datetime(df['TIME'], unit='s', origin='unix')
    # データフレームのインデックスを'timestamp'に設定
    df.set_index('timestamp', inplace=True)
    # 0.2ミリ秒間隔にリサンプリング
    df = df.resample('200us').mean()
    df = df.interpolate(method='linear')
    df = df.reset_index()

    l_INPUT = ['TIME',
               'MMotAct', 
               'p_MC_Model',
               'SasInCor', 
               'v_FL',
               'v_FR', 
               'v_RL',
               'v_RR',
               'MotPedalPosDriver', 
               'p_Model_FL',
               'p_Model_FR',
               'p_Model_RL',
               'p_Model_RR',
               'vGiF', 
               'ayToF',
               'Co_FA',
               'vRef_sum_ax',
               'vRef_sum_ax_ABS',
               'vRef_sum_ax_TCS',
               'nDrvUnit']

    if vRef.Check_Sigs(df, l_INPUT):
        df_input = df[l_INPUT]
        df_input.loc[:, 'TIME'] = df_input.loc[:, 'TIME'] + 10.001
    
        d_data = {}
        d_data['TIME'] = np.linspace(0, 10, 10001)
        d_data['MMotAct'] = 0
        d_data['p_MC_Model'] = 0
        d_data['SasInCor'] = np.concatenate((np.linspace(0, df.loc[0, 'SasInCor'], 8001), np.linspace(df.loc[0, 'SasInCor'], df.loc[0, 'SasInCor'], 2000)))
        d_data['v_FL'] = np.concatenate((np.linspace(0, df.loc[0, 'v_FL'], 8001), np.linspace(df.loc[0, 'v_FL'], df.loc[0, 'v_FL'], 2000)))
        d_data['v_FR'] = np.concatenate((np.linspace(0, df.loc[0, 'v_FR'], 8001), np.linspace(df.loc[0, 'v_FR'], df.loc[0, 'v_FR'], 2000)))
        d_data['v_RL'] = np.concatenate((np.linspace(0, df.loc[0, 'v_RL'], 8001), np.linspace(df.loc[0, 'v_RL'], df.loc[0, 'v_RL'], 2000)))
        d_data['v_RR'] = np.concatenate((np.linspace(0, df.loc[0, 'v_RR'], 8001), np.linspace(df.loc[0, 'v_RR'], df.loc[0, 'v_RR'], 2000)))
        d_data['MotPedalPosDriver'] = 100
        d_data['p_Model_FL'] = 0
        d_data['p_Model_FR'] = 0
        d_data['p_Model_RL'] = 0
        d_data['p_Model_RR'] = 0
        d_data['vGiF'] = 0
        d_data['ayToF'] = 0
        d_data['Co_FA'] = 0.5
        d_data['vRef_sum_ax'] = np.concatenate((np.linspace(0, df.loc[0, 'vRef_sum_ax'], 8001), np.linspace(df.loc[0, 'vRef_sum_ax'], df.loc[0, 'vRef_sum_ax'], 2000)))
        d_data['vRef_sum_ax_ABS'] = 10000.
        d_data['vRef_sum_ax_TCS'] = 10000.
        
        df_add = pd.DataFrame(d_data)
        
        df_input = Common.CONCAT_DATA([df_add, df_input])
    
        l_Sig = ['v_FL', 'v_FR', 'v_RL', 'v_RR', 'vRef_sum_ax', 'Co_FA']
    
        for Sig in l_Sig:
            df_input[Sig] = df_input[Sig].rolling(window=50, center=True).mean()
        
        df_input.to_csv(CSV_out, index=False)
    else:
        CSV_out = None
        
    return CSV_out


# In[12]:


# CSV = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\3BVT\\Test\\20240213_0001.csv'
# CSV_out = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\3BVT\\Test\\20240213_0001_out.csv'
# _ = MakeInputForEst(CSV, CSV_out)


# In[13]:


def EstimateWheelTorque_1(CSV, Add, Dir, Mdl_Est, eng, override, zip_est_file=False):
    OUT = None
    l_Remove = []

    file_name, file_extension = os.path.splitext(os.path.basename(CSV))
    OUT_override = f'{Dir}{file_name}{Add}.csv'

    if zip_est_file == True and os.path.exists(OUT_override) == True:
        _ = Common.SAVE_CSV_to_ZIP(OUT_override, remove_csv=True)
        
    root, ext = os.path.splitext(OUT_override)
    zip_file = f'{root}.zip'
    extracted_files = Common.Unpack_ZIP_to_CSV(zip_file, remove_zip=False)  

    if override == True or os.path.exists(OUT_override) == False:
        if os.path.exists(CSV):
            OUT_Est_Input = MakeInputForEst(CSV, f'{Dir}simin_Est_Input.csv')
    
            if os.path.exists(OUT_Est_Input):
                OUT_Est = RunMatlab(Mdl_Est, f'{Dir}simout_Est.csv', eng)
                l_Remove.append(OUT_Est_Input)
    
                if os.path.exists(OUT_Est):
                    # file_name, file_extension = os.path.splitext(os.path.basename(CSV))
                    # OUT = f'{Dir}{file_name}{Add}.csv'
                    OUT = OUT_override
                    
                    if os.path.exists(OUT):
                        os.remove(OUT)
                        
                    os.rename(OUT_Est, OUT)
                    
                    if not os.path.exists(OUT):
                        OUT = None
                    else:
                        _ = Common.SAVE_CSV_to_ZIP(OUT, remove_csv=True)

        
        for F in l_Remove:
            if os.path.exists(F):
                os.remove(F)

    if override == False and os.path.exists(OUT_override) == True:
        OUT = OUT_override

    for file in extracted_files:
        if os.path.exists(file):
            os.remove(file)

    return OUT


# In[14]:


def Check_estimated_vRef(file):
    df = pd.read_table(file, sep=",", index_col=None)
    if df['vRef_sum_ax_ABS'].min() != 10000. or df['vRef_sum_ax_TCS'].min() != 10000.:
        estimated = True
    else:
        estimated = False
    return estimated


# In[2]:


def EstimateWheelTorque(d_File, Add, Dir, Mdl, matlab_version, override=True):
    eng = matlab.engine.start_matlab(f"matlab.engine.shareEngine('{matlab_version}')")

    d_Result = {}

    for i, File in enumerate(d_File):
        if Check_estimated_vRef(File):
            try:
                Result = EstimateWheelTorque_1(File, f'_{str(i)}_{Add}', Dir, Mdl, eng, override)
            except matlab.engine.MatlabExecutionError:
                Result = None
        else:
            Result = None
        print(f'{i+1}/{len(d_File)}, {File}: {Result}')
        if Result != None:
            d_Result[Result] = d_File[File]

    eng.quit()
            
    return d_Result


# ### Run

# In[ ]:


# DIR = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\'
# MDL_Est = f'{DIR}Carsim_Est_3_HGT_3A0A.mdl'

# matlab_version = "R2022b"
# eng = matlab.engine.start_matlab(f"matlab.engine.shareEngine('{matlab_version}')")
# d_SIMs = EstimateWheelTorque(d_Selected_CSVs, '3A0A_ABS_FHEV_KA_4WD', DIR, MDL_Est, override=False)
# eng.quit()

# PKL = f'{DIR}\\Sim_List.pkl'
# with open(PKL, 'wb') as file:
#     pickle.dump(d_SIMs, file)


# # Data for ML

# ## Resampling-time from 0.2ms to 5ms

# In[22]:


def Resampling(d_Files, override=False, sampling_time='5ms'):
    for File in d_Files:
        root, ext = os.path.splitext(File)
        zip_file = f'{root}.zip'
        File_new = f'{root}_{sampling_time}.zip'
        
        if os.path.exists(File_new) == False or override == True:
            extracted_files = Common.Unpack_ZIP_to_CSV(zip_file, remove_zip=False)
            output = Common.Resampling_CSV(extracted_files[0], sampling_time='5ms')
            _ = Common.SAVE_CSV_to_ZIP(output, remove_csv=True)
            data = d_Files[File]
            data['Features'] = output
            d_Files[File] = data
    
            for file in extracted_files:
                if os.path.exists(file):
                    os.remove(file)
                    
        else:
            data = d_Files[File]
            data['Features'] = f'{root}_{sampling_time}.csv'
            d_Files[File] = data

    return d_Files


# ## Selected_Data

# ## Features

# In[23]:


def Create_Features(d_Datas, abrollumfang, vibration_thresholds=None, vibration=False):
    for F in d_Datas:
        File = d_Datas[F]['Features']
        root, ext = os.path.splitext(File)
        zip_file = f'{root}.zip'
        extracted_files = Common.Unpack_ZIP_to_CSV(zip_file, remove_zip=True)
        df = pd.read_table(extracted_files[0], sep=',')

        df['diff_Vx_L1'] = df['Vx_L1'] - df['Vx_R1']
        df['diff_Vx_R1'] = df['Vx_R1'] - df['Vx_L1']
        df['diff_Vx_L2'] = df['Vx_L2'] - df['Vx_R2']
        df['diff_Vx_R2'] = df['Vx_R2'] - df['Vx_L2']
        df['dev_Vx_L1'] = df['Vx_L1'] - df['Vx']
        df['dev_Vx_R1'] = df['Vx_R1'] - df['Vx'] 
        df['dev_Vx_L2'] = df['Vx_L2'] - df['Vx']
        df['dev_Vx_R2'] = df['Vx_R2'] - df['Vx']

        df['abs_diff_Vx'] = (df['Vx'] - df['V'] * 3.6).abs()
        df['abs_diff_Vx_L1'] = (df['Vx_L1'] - df['v_FL'] * 3.6).abs()
        df['abs_diff_Vx_R1'] = (df['Vx_R1'] - df['v_FR'] * 3.6).abs()
        df['abs_diff_Vx_L2'] = (df['Vx_L2'] - df['v_RL'] * 3.6).abs()
        df['abs_diff_Vx_R2'] = (df['Vx_R2'] - df['v_RR'] * 3.6).abs()

        df['GearRatio'] = (df['nDrvUnit'] / 60 ) / (((df['Vx_L1'] + df['Vx_R1']) / 2 * df['Co_FA'] + (df['Vx_L2'] + df['Vx_R2']) / 2 * (1 - df['Co_FA'])) / 3.6 / abrollumfang)
        df['GearRatio'] = df['GearRatio'].clip(lower=0, upper=20)

        df['abs_Sas'] = df['Steer_in'].abs()
        df['abs_vGiF'] = df['vGiF'].abs()
        df['abs_ayToF'] = df['ayToF'].abs()
    
        if 'ThrottlePos' not in df.columns and 'Throttle_in' in df.columns:
            df['ThrottlePos'] = df['Throttle_in']
        
        Filtered_Sigs = [(['IMP_FX_L1', 'IMP_FX_R1', 'IMP_FX_L2', 'IMP_FX_R2'], [40], True),
                         (['IMP_MY_OUT_DR_L1', 'IMP_MY_OUT_DR_R1', 'IMP_MY_OUT_DR_L2', 'IMP_MY_OUT_DR_R2'], [40], True),
                         (['Fz_L1', 'Fz_R1', 'Fz_L2', 'Fz_R2'], [400], True),
                         (['Ax_SM', 'diff_Vx_L1', 'diff_Vx_R1', 'dev_Vx_L1', 'dev_Vx_R1', 'diff_Vx_L2', 'diff_Vx_R2', 'dev_Vx_L2', 'dev_Vx_R2', 'MMotAct', 'ThrottlePos', 'GearRatio', 'p_FL', 'p_FR'], [10, 20, 40, 100, 200, 400], False)]
        
        for data in Filtered_Sigs:
            Sigs, Ns, center = data
            for Sig in Sigs:
                for N in Ns:
                    df[f'{Sig}_{str(N)}'] = df[Sig].rolling(window=N, center=center).mean()

        if vibration == True:
            df = detect_vibration(df, thresholds)
        
        df.to_csv(extracted_files[0], index=False)
        _ = Common.SAVE_CSV_to_ZIP(extracted_files[0], remove_csv=True)


# In[24]:


def detect_vibration(df, thresholds):
    values = ['Vx_L1', 'Vx_R1',
              'Vx_L2', 'Vx_R2', 
              'IMP_FX_L1', 'IMP_FX_R1',
              'IMP_FX_L2', 'IMP_FX_R2', 
              'IMP_MY_OUT_DR_L1', 'IMP_MY_OUT_DR_R1', 
              'IMP_MY_OUT_DR_L2', 'IMP_MY_OUT_DR_R2']
    
    window_size = 10
    rolling_means = {}
    rolling_stds = {}
    vibrations = {}
    
    for value in values:
        rolling_means[f'{value}_rolling_mean'] = df[value].rolling(window=window_size).mean()
        rolling_stds[f'{value}_rolling_std'] = df[value].rolling(window=window_size).std()
    
    rolling_means_df = pd.DataFrame(rolling_means)
    rolling_stds_df = pd.DataFrame(rolling_stds)
    
    # 重複する場合は上書きする
    df = df.drop(columns=[col for col in rolling_means_df.columns if col in df.columns])
    df = df.drop(columns=[col for col in rolling_stds_df.columns if col in df.columns])
    
    df = pd.concat([df, rolling_means_df, rolling_stds_df], axis=1)
    
    for value in values:
        threshold = thresholds[f'{value}_rolling_std']
        vibrations[f'{value}_vibration'] = np.abs(df[value] - df[f'{value}_rolling_mean']) > threshold
        
    vibrations_df = pd.DataFrame(vibrations)
    
    # 重複する場合は上書きする
    df = df.drop(columns=[col for col in vibrations_df.columns if col in df.columns])
    
    df = pd.concat([df, vibrations_df], axis=1)
    
    detects = []
    counter = 0
    
    for i in df.index:
        vibration = False
        for value in values:
            if df.loc[i, f'{value}_vibration']:
                vibration = True
                break
    
        if vibration:
            counter = 0.05
        else:
            counter -= 0.005
            counter = max(counter, 0)
            
        if counter > 0 and df.loc[i, 'Vx'] > 2.0:
            detects.append(1)
        else:
            detects.append(0)
    
    detects_series = pd.Series(detects, name='vibration')
    
    # 重複する場合は上書きする
    if 'vibration' in df.columns:
        df = df.drop(columns=['vibration'])
    
    df = pd.concat([df, detects_series], axis=1)

    return df


# In[25]:


def detect_vibration_org(df, thresholds):
    values = ['Vx_L1', 'Vx_R1',
              'Vx_L2', 'Vx_R2', 
              'IMP_FX_L1', 'IMP_FX_R1',
              'IMP_FX_L2', 'IMP_FX_R2', 
              'IMP_MY_OUT_DR_L1', 'IMP_MY_OUT_DR_R1', 
              'IMP_MY_OUT_DR_L2', 'IMP_MY_OUT_DR_R2']
    
    # 移動平均と移動標準偏差を計算
    stds = {}
    window_size = 10
    for value in values:
        df[f'{value}_rolling_mean'] = df[value].rolling(window=window_size).mean()
        df[f'{value}_rolling_std'] = df[value].rolling(window=window_size).std()
    
        # 振動の閾値を設定
        threshold = thresholds[f'{value}_rolling_std']
    
        # 振動を検出
        df[f'{value}_vibration'] = np.abs(df[value] - df[f'{value}_rolling_mean']) > threshold
    
    detects = []
    counter = 0
    
    for i in df.index:
        vibration = False
        for value in values:
            if df.loc[i, f'{value}_vibration'] == True:
                vibration = True
                break
    
        if vibration == True:
            counter = 0.05
        else:
            counter -= 0.005
            counter = max(counter, 0)
            
        if counter > 0 and df.loc[i, 'Vx'] > 2.0:
            detects.append(1)
        else:
            detects.append(0)
    
    # detectsをシリーズに変換し、名前を付ける
    detects_series = pd.Series(detects, name='vibration')
    
    # dfとdetects_seriesを連結
    df = pd.concat([df, detects_series], axis=1)

    return df


# In[26]:


def detect_vibration_threshold(df):
    values = [('Vx_L1', 8.0), ('Vx_R1', 8.0),
              ('Vx_L2', 8.0), ('Vx_R2', 8.0),
              ('IMP_FX_L1', 1.0), ('IMP_FX_R1', 1.0),
              ('IMP_FX_L2', 1.0), ('IMP_FX_R2', 1.0),
              ('IMP_MY_OUT_DR_L1', 1.0), ('IMP_MY_OUT_DR_R1', 1.0), 
              ('IMP_MY_OUT_DR_L2', 8.0), ('IMP_MY_OUT_DR_R2', 8.0)]
    
    # 移動平均と移動標準偏差を計算
    stds = {}
    window_size = 10
    for data in values:
        value, gain = data
        df[f'{value}_rolling_mean'] = df[value].rolling(window=window_size).mean()
        df[f'{value}_rolling_std'] = df[value].rolling(window=window_size).std()
    
        # 振動の閾値を設定
        threshold = gain * df[f'{value}_rolling_std'].mean()
        stds[f'{value}_rolling_std'] = threshold
    
        # 振動を検出
        df[f'{value}_vibration'] = np.abs(df[value] - df[f'{value}_rolling_mean']) > threshold
    
    detects = []
    counter = 0
    
    for i in df.index:
        vibration = False
        for data in values:
            value, _ = data
            if df.loc[i, f'{value}_vibration'] == True:
                vibration = True
                break
    
        if vibration == True:
            counter = 0.05
        else:
            counter -= 0.005
            counter = max(counter, 0)
            
        if counter > 0 and df.loc[i, 'Vx'] > 2.0:
            detects.append(1)
        else:
            detects.append(0)
    
    # detectsをシリーズに変換し、名前を付ける
    detects_series = pd.Series(detects, name='vibration')
    
    # dfとdetects_seriesを連結
    df = pd.concat([df, detects_series], axis=1)

    return df, stds


# ## Adding_Singals

# In[27]:


def Adding_Signals(d_datas, add_signals, plt, sampling=0.005):
    d_updated_datas = {}

    d_plt = Common.Select_Signal(plt)
    
    for label in d_datas:
        zip = d_datas[label]['ZIP']
        csv, NotConvert = Common.DataTreatment(zip, d_plt, sampling)
        target_csv = d_datas[label]['Features']
        root, ext = os.path.splitext(target_csv)
        target_zip = f'{root}.zip'
        _ = Common.Unpack_ZIP_to_CSV(target_zip, remove_zip=False)

        if NotConvert == False:               
            update = Adding_Siganls_w_Initial(csv, target_csv, add_signals)
            if update == True:
                d_updated_datas[file] = d_datas[file]
    
    return d_updated_datas


def Adding_Siganls_w_Initial(csv, target_csv, add_sigs):
    updated = True
    
    df = pd.read_table(csv, sep=',')
    df_target = pd.read_table(target_csv, sep=',') 

    for sig in add_sigs:
        if sig in df.columns:
            data = list(df[sig])
            n = int(10000 / 5  + 1)
            updated_data =  list(np.linspace(0, 10, n)) + list(data)
            # print(len(df_target), len(updated_data), len(list(data)), len(list(np.linspace(0, 10, n))))
            df_target[sig] = updated_data
        else:
            updated = False

    if updated == True:
        df_target.to_csv(target_csv, index=False)
        _ = Common.SAVE_CSV_to_ZIP(target_csv, remove_csv=True)
    
    return updated_data, updated


# ## Create_Train_Val_Data

# In[28]:


def Create_Train_Val_Data(d_Datas, dir, train=0.7):
    data_list = list(d_Datas.keys())
    shuffled_indices = np.random.permutation(len(data_list))
    end = int(len(data_list) * train)
    train_indices = shuffled_indices[:end]
    val_indices = shuffled_indices[end:]

    train_list = []
    for i in train_indices:
        file = d_Datas[data_list[i]]['Features']
        train_list.append(file)

    df = Create_Data(train_list)
    train_file = f'{dir}train_data.pkl'
    with open(train_file, 'wb') as file:
        pickle.dump(df, file)
        
    val_list = []
    for i in val_indices:
        file = d_Datas[data_list[i]]['Features']
        val_list.append(file)

    df = Create_Data(val_list)

    val_file = f'{dir}val_data.pkl'
    with open(val_file, 'wb') as file:
        pickle.dump(df, file)

    return train_file, val_file, train_list, val_list


# In[29]:


def Create_Data(data_list):    
    for i, File in enumerate(data_list):    
        root, ext = os.path.splitext(File)
        zip_file = f'{root}.zip'
        extracted_files = Common.Unpack_ZIP_to_CSV(zip_file, remove_zip=False)
        df = pd.read_table(extracted_files[0], sep=',')
        
        if i == 0:
            df_concat = df
        else:
            df_concat = pd.concat([df_concat, df], axis=0, ignore_index=True)
    
        for file in extracted_files:
            if os.path.exists(file):
                os.remove(file)

    # print(File)
    change_sigs = {}
    change_sigs['IMP_FX_x1'] = ['IMP_FX_L1_40', 'IMP_FX_R1_40']
    change_sigs['IMP_FX_x2'] = ['IMP_FX_L2_40', 'IMP_FX_R2_40']
    change_sigs['IMP_MY_OUT_DR_x1'] = ['IMP_MY_OUT_DR_L1_40', 'IMP_MY_OUT_DR_R1_40']
    change_sigs['IMP_MY_OUT_DR_x2'] = ['IMP_MY_OUT_DR_L2_40', 'IMP_MY_OUT_DR_R2_40']
    change_sigs['Fz_x1'] = ['Fz_L1_400', 'Fz_R1_400']
    change_sigs['Fz_x2'] = ['Fz_L2_400', 'Fz_R2_400']
    change_sigs['diff_Vx_x1_10'] = ['diff_Vx_L1_10', 'diff_Vx_R1_10']
    change_sigs['diff_Vx_x1_20'] = ['diff_Vx_L1_20', 'diff_Vx_R1_20']
    change_sigs['diff_Vx_x1_40'] = ['diff_Vx_L1_40', 'diff_Vx_R1_40']
    change_sigs['diff_Vx_x1_100'] = ['diff_Vx_L1_100', 'diff_Vx_R1_100']
    change_sigs['diff_Vx_x1_200'] = ['diff_Vx_L1_200', 'diff_Vx_R1_200']
    change_sigs['diff_Vx_x1_400'] = ['diff_Vx_L1_400', 'diff_Vx_R1_400']
    change_sigs['dev_Vx_x1_10'] = ['dev_Vx_L1_10', 'dev_Vx_R1_10']
    change_sigs['dev_Vx_x1_20'] = ['dev_Vx_L1_20', 'dev_Vx_R1_20']
    change_sigs['dev_Vx_x1_40'] = ['dev_Vx_L1_40', 'dev_Vx_R1_40']
    change_sigs['dev_Vx_x1_100'] = ['dev_Vx_L1_100', 'dev_Vx_R1_100']
    change_sigs['dev_Vx_x1_200'] = ['dev_Vx_L1_200', 'dev_Vx_R1_200']
    change_sigs['dev_Vx_x1_400'] = ['dev_Vx_L1_400', 'dev_Vx_R1_400']
    change_sigs['diff_Vx_x2_10'] = ['diff_Vx_L2_10', 'diff_Vx_R2_10']
    change_sigs['diff_Vx_x2_20'] = ['diff_Vx_L2_20', 'diff_Vx_R2_20']
    change_sigs['diff_Vx_x2_40'] = ['diff_Vx_L2_40', 'diff_Vx_R2_40']
    change_sigs['diff_Vx_x2_100'] = ['diff_Vx_L2_100', 'diff_Vx_R2_100']
    change_sigs['diff_Vx_x2_200'] = ['diff_Vx_L2_200', 'diff_Vx_R2_200']
    change_sigs['diff_Vx_x2_400'] = ['diff_Vx_L2_400', 'diff_Vx_R2_400']
    change_sigs['dev_Vx_x2_10'] = ['dev_Vx_L2_10', 'dev_Vx_R2_10']
    change_sigs['dev_Vx_x2_20'] = ['dev_Vx_L2_20', 'dev_Vx_R2_20']
    change_sigs['dev_Vx_x2_40'] = ['dev_Vx_L2_40', 'dev_Vx_R2_40']
    change_sigs['dev_Vx_x2_100'] = ['dev_Vx_L2_100', 'dev_Vx_R2_100']
    change_sigs['dev_Vx_x2_200'] = ['dev_Vx_L2_200', 'dev_Vx_R2_200']
    change_sigs['dev_Vx_x2_400'] = ['dev_Vx_L2_400', 'dev_Vx_R2_400']
    change_sigs['p_Fx_10'] = ['p_FL_10', 'p_FR_10']
    change_sigs['p_Fx_20'] = ['p_FL_20', 'p_FR_20']
    change_sigs['p_Fx_40'] = ['p_FL_40', 'p_FR_40']
    change_sigs['p_Fx_100'] = ['p_FL_100', 'p_FR_100']
    change_sigs['p_Fx_200'] = ['p_FL_200', 'p_FR_200']
    change_sigs['p_Fx_400'] = ['p_FL_400', 'p_FR_400']
    
    for i in [0, 1]:
        for sig in change_sigs:
            # print(df_concat.columns)
            # print(change_sigs[sig][i])
            df_concat[sig] = df_concat[change_sigs[sig][i]]
        if i == 0:
            df_out = df_concat
        else:
            df_out = pd.concat([df_out, df_concat], axis=0, ignore_index=True)
    
    # df_out.to_csv(out, index=False)
    return df_out


# ## Run

# In[ ]:


# # pickleファイルを読み込む
# file_path = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\Sim_List.pkl'
# with open(file_path, 'rb') as file:
#     d_SIMs = pickle.load(file)

# # サンプリング時間の変更
# d_SIMs = Resampling(d_SIMs, override=False)
# print(len(d_SIMs))


# In[ ]:


# 特徴量（振動検出）の検討
# Dir = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\'
# File = f'{Dir}20210203_0009_1434_3A0A_ABS_FHEV_KA_4WD_5ms.csv'
# df = pd.read_table(File, sep=',')
# df, thresholds = detect_vibration_threshold(df)
# df.to_csv(f'{Dir}vibration_threshold.csv', index=False)
# print(thresholds)

# File = f'{Dir}20200114_0033_2521_3A0A_ABS_FHEV_KA_4WD_5ms.csv'
# df = pd.read_table(File, sep=',')
# df = detect_vibration(df, thresholds)
# df.to_csv(f'{Dir}vibration.csv', index=False)

# 特徴量の作成
# Vibration_Thresholds = {'Vx_L1_rolling_std': 5.545621794003921,
#                        'Vx_R1_rolling_std': 4.5422820543420155,
#                        'Vx_L2_rolling_std': 0.5126707363352918,
#                        'Vx_R2_rolling_std': 0.5143928413297248,
#                        'IMP_FX_L1_rolling_std': 815.7665907984712,
#                        'IMP_FX_R1_rolling_std': 815.7665907984712,
#                        'IMP_FX_L2_rolling_std': 0.012278214377030465,
#                        'IMP_FX_R2_rolling_std': 0.012278214377030465,
#                        'IMP_MY_OUT_DR_L1_rolling_std': 320.3087461357675,
#                        'IMP_MY_OUT_DR_R1_rolling_std': 301.498065560067,
#                        'IMP_MY_OUT_DR_L2_rolling_std': 241.34109167283475,
#                        'IMP_MY_OUT_DR_R2_rolling_std': 233.73903668765968}

# Abrollumfang = 2.215
# Create_Features(d_SIMs, Abrollumfang)


# In[ ]:


# Adding signals
# PLT = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Tire_Torque\\Conv_Est_Tire.PLT'
# Add_Signals = ['MDriverRequest']
# d_SIMs = Adding_Signals(d_SIMs, Add_Signals, PLT)


# In[ ]:


# データ選別
# contain_list = ['hev']
# not_contain_list = ['pet']
# ignore_list = []
# d_SIMs = Search_Comment(d_SIMs, contain_list, not_contain_list, ignore_list=ignore_list)
# print(f'HEV: {len(d_SIMs)}')


# In[ ]:


# データチェック
# # Coping target file
# for label in d_SIMs:
#     root, ext = os.path.splitext(label)
#     # print(f'{root}_5ms.zip')

#     # コピー元ファイルのパス
#     source_path = f'{root}_5ms.zip'
#     basename = os.path.basename(source_path)

#     # コピー先ファイルのパス
#     dir = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\Check\\'
#     destination_path = f'{dir}{basename}'

#     if os.path.exists(source_path):
#         # ファイルをコピーする
#         shutil.copyfile(source_path, destination_path)
        
#         print(f"File copied from {source_path} to {destination_path}")


# In[ ]:


# 特徴量の作成
# Dir = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\'
# Train_Data, Val_Data, Train_list, Val_list = Create_Train_Val_Data(d_SIMs, Dir, train=0.8)
# print(f'Train: {Train_Data}')
# print(f'Validation: {Val_Data}')

# PKL = f'{Dir}\\Train_Val_Datas.pkl'
# with open(PKL, 'wb') as file:
#     pickle.dump((Train_Data, Val_Data, Train_list, Val_list), file)


# # ML

# ## MyDataset

# In[15]:


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


# In[16]:


class CustomNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


# ## Run

# ### IMP_MY_OUT_DR_x1

# In[17]:


def Treatment_Data(df, selected_columns, target):
    df = df[(df['TIME'] > 10) & (df['ThrottlePos'] > 0) & (df['V_TCS'] != 10000)]

    df = df[(df['abs_diff_Vx'] < 1.0) & (df['abs_diff_Vx_L1'] < 1.0) & (df['abs_diff_Vx_R1'] < 1.0)
            & (df['abs_diff_Vx_L2'] < 1.0) & (df['abs_diff_Vx_R2'] < 1.0)]
    
    # df = df[(df['diff_Vx_L1_10'] < 1.0) & (df['diff_Vx_R1_10'] < 1.0) 
    #         & (df['diff_Vx_L2_10'] < 1.0) & (df['diff_Vx_R2_10'] < 1.0)]

    df = df[(df['abs_Sas'] < 0.5) & (df['abs_vGiF'] < 0.5) & (df['abs_ayToF'] < 0.5)]

    df = df[(df['GearRatio_10'] < 20) & (df['Ax_SM_40'] > 0.01)]
    
    condition = (df['ThrottlePos'] > 90) & (df['dev_Vx_x1_10'] < 0) & (df['dev_Vx_x1_20'] < 0) & (df['dev_Vx_x1_40'] < 0)
    df = df[~condition]
   
    df = df[selected_columns]
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y


# In[ ]:


# # データ準備
# Dir = 'c:\\TSDE_Workarea\\ktt2yk\\CARSIM\\CarSim_HGT_3A0A_BB85363_Var10_20230516_Matlab2022b\\'
# train_data = f'{Dir}train_data.pkl'
# val_data = f'{Dir}val_data.pkl'

# Features = ['Vx', 
#             'Ax_SM_10', 'Ax_SM_20', 'Ax_SM_40', 'Ax_SM_100', 'Ax_SM_200', 'Ax_SM_400', 
#             'diff_Vx_x1_10', 'diff_Vx_x1_20', 'diff_Vx_x1_40', 'diff_Vx_x1_100', 'diff_Vx_x1_200', 'diff_Vx_x1_400',
#             'dev_Vx_x1_10', 'dev_Vx_x1_20', 'dev_Vx_x1_40', 'dev_Vx_x1_100', 'dev_Vx_x1_200', 'dev_Vx_x1_400',
#             'p_Fx_10', 'p_Fx_20', 'p_Fx_40', 'p_Fx_100', 'p_Fx_200', 'p_Fx_400',
#             # 'GearRatio_10', 'GearRatio_20', 'GearRatio_40', 'GearRatio_100', 'GearRatio_200', 'GearRatio_400',
#             'MMotAct', 'MMotAct_10', 'MMotAct_20', 'MMotAct_40', 'MMotAct_100', 'MMotAct_200', 'MMotAct_400',
#             'ThrottlePos', 'ThrottlePos_10', 'ThrottlePos_20', 'ThrottlePos_40', 'ThrottlePos_100', 'ThrottlePos_200', 'ThrottlePos_400', 
#             'IMP_FX_x1', 'IMP_MY_OUT_DR_x1']
    
# Target = 'IMP_MY_OUT_DR_x1'

# print(f'Read: {train_data}')
# with open(train_data, 'rb') as file:
#     df = pickle.load(file)
# X, y = Treatment_Data(df, Features, Target)

# print(f'Read: {val_data}')
# with open(val_data, 'rb') as file:
#     df = pickle.load(file)
# X_val, y_val = Treatment_Data(df, Features, Target)

# # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = X, X_val, y, y_val

# # Numpy配列をTensorに変換
# X_train = torch.tensor(X_train.values, dtype=torch.float32)
# y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
# X_val = torch.tensor(X_val.values, dtype=torch.float32)
# y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# # データの平均と標準偏差を計算
# mean = X_train.mean(dim=0)
# std = X_train.std(dim=0)

# # Normalize変換を定義
# transform = CustomNormalize(mean, std)

# train_dataset = MyDataset(X_train, y_train, transform)
# val_dataset = MyDataset(X_val, y_val, transform)

# train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=0)

# # モデル準備
# class NN(nn.Module):
#     def __init__(self, num_in):
#         super().__init__()
#         self.l1 = nn.Linear(num_in, 256)
#         self.b1 = nn.BatchNorm1d(256)
#         self.l2 = nn.Linear(256, 128)
#         self.b2 = nn.BatchNorm1d(128)
#         self.l3 = nn.Linear(128, 64)
#         self.b3 = nn.BatchNorm1d(64)
#         self.l4 = nn.Linear(64, 32)
#         self.b4 = nn.BatchNorm1d(32)
#         self.l5 = nn.Linear(32, 1)
        
#         # パラメータの初期化
#         self._initialize_weights()

#     def forward(self, x):
#         x = torch.relu(self.b1(self.l1(x)))
#         x = torch.relu(self.b2(self.l2(x)))
#         x = torch.relu(self.b3(self.l3(x)))
#         x = torch.relu(self.b4(self.l4(x)))
#         x = self.l5(x)
#         return x
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
        
# X, y = next(iter(train_loader))
# num_in = X.shape[1]

# model = NN(num_in)
# model = model.to(device)
# lrs, losses = utils.lr_finder(model, train_loader, nn.MSELoss(), lr_multiplier=1.1)

# fig = plt.figure(figsize=(8, 3), dpi=100)
# plt.plot(lrs, losses, label='losses')
# plt.xscale('log')
# plt.legend()
# plt.ylim(0, 1000000)
# plt.show()

# model = NN(num_in)
# model = model.to(device)

# num_epoch = 100
# early_stopping = 20
# save_path='IMP_MY_OUT_DR_x1_model.pth'
# # scheduler=None
# # scheduler = StepLR(opt, step_size=30, gamma=0.1)
# # opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# opt = optim.SGD(model.parameters(), lr=0.001)
# scheduler = CosineAnnealingLR(opt, T_max=20)
# train_losses, val_losses, val_accuracies = utils.learn(model, train_loader, val_loader, opt, nn.MSELoss(), num_epoch=num_epoch, early_stopping=early_stopping, save_path=save_path, scheduler=scheduler)

# # データ準備
# print(f'Read: {val_data}')
# with open(val_data, 'rb') as file:
#     df = pickle.load(file)
# X_val, y_val = Treatment_Data(df, Features, Target)

# # Numpy配列をTensorに変換
# X = torch.tensor(X_val.values, dtype=torch.float32)
# y = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
# X = transform(X)

# save_path='IMP_MY_OUT_DR_x1_model.pth'
# loaded_model = torch.load(save_path)
# learned_model = NN(num_in)
# learned_model.load_state_dict(loaded_model['model_state_dict'])

# y_pred = learned_model(X)

# # MSEの計算
# # loss = mean_squared_error(y_val.detach().numpy(), y_pred.detach().numpy())
# mse = nn.MSELoss()
# loss = mse(y, y_pred)
# print(f'MSE Loss: {loss}')

# # fig = plt.figure(figsize=(8, 3), dpi=100)
# # plt.plot(x, y, label='true')
# # plt.plot(x, y_pred, label='pred')
# # plt.legend()
# # plt.ylim(0, 1000000)

# X_val[f'{Target}'] = y.detach().numpy()
# X_val[f'{Target}_pred'] = y_pred.detach().numpy()
# X_val['TIME'] = np.arange(0, len(X_val)*0.005, 0.005)
# columns = ['TIME'] + list(X_val.columns)
# X_val = X_val[columns]
# X_val.to_csv(f'{Target}_pred.csv', index=False)

