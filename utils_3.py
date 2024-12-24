#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import Common_2 as Common


# # ABS

# ## ABS_Braking_List

# In[2]:


def ABS_Braking_List(df, sampling):
    # bls = []
    bls_counter = []
    bls_off_counter = []
    bls_index = []
    counter = 0
    off_counter = 0
    index_start = None
    p_MC_Bls = 0
    Bls_01 = 0
    
    for n, i in enumerate(df.index):       
        Bls1 = df.loc[i, 'BlsAsw']
        p_MC = df.loc[i, 'p_MC_Model']
        p_FL = df.loc[i, 'p_Model_FL']
        p_FR = df.loc[i, 'p_Model_FR']
        p_RL = df.loc[i, 'p_Model_RL']
        p_RR = df.loc[i, 'p_Model_RR']

        if i == 0:
            Bls0 = Bls1

        if Bls0 == 0 and Bls1 == 1:
            Bls_01 = 1

        if Bls1 == 0:
            Bls_01 = 0
        
        if p_MC > 10 and (p_FL > 10 or p_FR > 10 or p_RL > 10 or p_RR > 10):
            p_MC_Bls = 1

        if p_MC == 0:
            p_MC_Bls = 0
        
        if Bls_01 == 1 and p_MC_Bls == 1:
            counter += sampling
        else:
            counter = 0
            
        bls_counter.append(counter)
        bls_off_counter.append(off_counter)

        if Bls1 == 0:
            off_counter += sampling
        else:
            off_counter = 0

        if counter == sampling:
            index_start = i
        elif counter == 0 and off_counter > 1 and index_start != None:
            bls_index.append((index_start, i))
            index_start = None

        Bls0 = Bls1
        # bls.append(Bls)

    if index_start != None:
        bls_index.append((index_start, i))
    
    df['Bls_Counter'] = bls_counter
    df['Bls_Off_Counter'] = bls_off_counter
    # df['Bls'] = bls
    
    ABS_Ends = []
    index_ABS_Ends = []

    df['ax.F'] = df['ax'].rolling(window=20, center=True).mean()
    df['ax.diff'] = df['ax'].diff().abs().rolling(window=20, center=True).mean()
    
    for n, i in enumerate(df.index):
        # Bls = df.loc[i, 'Bls']
        PedalPos = df.loc[i, 'MotPedalPosDriver']
        ax = df.loc[i, 'ax.F']
        ax_diff = df.loc[i, 'ax.diff']
        # ABS = df.loc[i, 'ABS_active_120ms']
        Bls_Counter = df.loc[i, 'Bls_Counter']
        Bls_Off_Counter = df.loc[i, 'Bls_Off_Counter']
        v_FL = df.loc[i, 'v_FL']
        v_FR = df.loc[i, 'v_FR']
        v_RL = df.loc[i, 'v_RL']
        v_RR = df.loc[i, 'v_RR']

        v_Min = min([v_FL, v_FR, v_RL, v_RR])
        v_Max = max([v_FL, v_FR, v_RL, v_RR])
    
        if n == 0:
            ax0 = ax
            cross_reset = True
    
        if Bls_Counter == 0 and PedalPos == 0:
            cross_reset = True 
    
        if (ax > 0) and (ax * ax0 < 0) and cross_reset and (Bls_Counter > 1) and ax_diff > 0.05:
            ABS_Ends.append(1)
            index_ABS_Ends.append(i)
            cross_reset = False
        elif cross_reset == True and Bls_Off_Counter > 0.5 and v_Max - v_Min < 0.2:
            ABS_Ends.append(1)
            index_ABS_Ends.append(i)
            cross_reset = False            
        else:
            ABS_Ends.append(0)
    
        ax0 = ax
        
    df['ABS_End'] = ABS_Ends

    braking_list = []
    for indexs in bls_index:
        i_start, i_end = indexs
        df_range = df.iloc[i_start:i_end]
        if len(df_range[df_range['ABS_End'] == 1]) > 0:
            i_end_updated = df_range[df_range['ABS_End'] == 1].index[0]
            braking_list.append((i_start, i_end_updated))

    return df, braking_list


# ## ABS_AxSum

# In[3]:


def ABS_AxSum(df_all, braking_list, sampling): 
    data = []
    
    for indexs in braking_list:
        i_start, i_end = indexs
        df = df_all.iloc[i_start:i_end]
        
        # gains = []
        # losses = []
        best_loss = 10000
        best_gain = None
        best_offset = None
    
        for o in range(-20, 20, 5):
            offset = o / 100
            
            for g in range(-100, 100, 20):
                gain = 1 + g / 1000
                loss = 0
                
                for n, i in enumerate(df.index):
                    # Bls = df.loc[i, 'Bls']
                    PedalPos = df.loc[i, 'MotPedalPosDriver']
                    ax = df.loc[i, 'ax']
                    v_FL = df.loc[i, 'v_FL']
                    v_FR = df.loc[i, 'v_FR']
                    v_RL = df.loc[i, 'v_RL']
                    v_RR = df.loc[i, 'v_RR']
                    Bls_Counter = df.loc[i, 'Bls_Counter']
                    # cross_ax = df.loc[i, 'cross_ax']
                
                    if n == 0:
                        vx = (v_FL + v_FR + v_RL + v_RR) / 4
                    # if Bls_Counter == 0 and PedalPos == 0:
                    #     vx = (v_FL + v_FR + v_RL + v_RR) / 4
                    else:
                        vx += (ax + offset) * sampling * gain      
                
                loss = abs((v_FL + v_FR + v_RL + v_RR) / 4 - vx)
        
                if best_loss > loss:
                    best_loss = loss
                    best_gain = gain
                    best_offset = offset
        
        sum_ax = []
        
        for n, i in enumerate(df.index):
            # Bls = df.loc[i, 'Bls']
            PedalPos = df.loc[i, 'MotPedalPosDriver']
            ax = df.loc[i, 'ax']
            v_FL = df.loc[i, 'v_FL']
            v_FR = df.loc[i, 'v_FR']
            v_RL = df.loc[i, 'v_RL']
            v_RR = df.loc[i, 'v_RR']
            Bls_Counter = df.loc[i, 'Bls_Counter']
            # cross_ax = df.loc[i, 'cross_ax']
        
            if n == 0:
                vx = (v_FL + v_FR + v_RL + v_RR) / 4
        
            if Bls_Counter == 0 and PedalPos == 0:
                vx = (v_FL + v_FR + v_RL + v_RR) / 4
            else:
                vx += (ax + best_offset) * sampling * best_gain      
        
            vx = max(vx, -0.1)
            sum_ax.append(vx)
        
        # df['sum_ax'] = sum_ax
        # df['vRef_sum_ax'] = sum_ax
        df.loc[:, 'vRef_sum_ax_ABS'] = sum_ax
        df.loc[:, 'vRef_sum_ax'] = sum_ax
        df_all.update(df)

    return df_all


# # TCS

# ## TCS_Throttle_List

# In[45]:


def TCS_Throttle_List(df, sampling):
    pw_counters = []
    pw_counters_off = []
    pw_indexs = []
    counter = 0
    counter_off = 0
    index_start = None
    p_MC_Bls = 0
    MotPedalPosDriver_01 = None

    for n, i in enumerate(df.index):   
        MotPedalPosDriver1 = df.loc[i, 'MotPedalPosDriver']

        if MotPedalPosDriver1 == 0:
            counter_off += sampling
        else:
            counter_off = 0
            
        if i == 0:
            MotPedalPosDriver0 = MotPedalPosDriver1

        if MotPedalPosDriver0 == 0 and MotPedalPosDriver1 > 0:
            MotPedalPosDriver_01 = 1
            

        if MotPedalPosDriver_01 != None and MotPedalPosDriver1 == 0 and counter_off > 1:
            MotPedalPosDriver_01 = 0
        
        if MotPedalPosDriver_01 == 1:
            counter += sampling
        elif MotPedalPosDriver_01 == 0:
            if counter_off < 2.0:
                counter += sampling
            else:
                counter = 0
            
        pw_counters.append(counter)
        pw_counters_off.append(counter_off)

        if counter == sampling:
            index_start = i
        elif counter == 0 and index_start != None:
            pw_indexs.append((index_start, i))
            index_start = None

        MotPedalPosDriver0 = MotPedalPosDriver1
        # bls.append(Bls)

    if index_start != None:
        pw_indexs.append((index_start, i))
    
    df['PW_Counter'] = pw_counters
    df['PW_Counter_Off'] = pw_counters_off
    # print(f'pw_indexs: {pw_indexs}')
    
    ends= []
    
    for n, i in enumerate(df.index):
        PedalPos = df.loc[i, 'MotPedalPosDriver']
        PW_Counter = df.loc[i, 'PW_Counter']
        PW_Counter_Off = df.loc[i, 'PW_Counter_Off']
        p_MC = df.loc[i, 'p_MC_Model']
        p_FL = df.loc[i, 'p_Model_FL']
        p_FR = df.loc[i, 'p_Model_FR']
        p_RL = df.loc[i, 'p_Model_RL']
        p_RR = df.loc[i, 'p_Model_RR']
        v_FL = df.loc[i, 'v_FL']
        v_FR = df.loc[i, 'v_FR']
        v_RL = df.loc[i, 'v_RL']
        v_RR = df.loc[i, 'v_RR']

        v_Min = min([v_FL, v_FR, v_RL, v_RR])
        v_Max = max([v_FL, v_FR, v_RL, v_RR])
    
        if PedalPos == 0 and (p_FL > 5) and (p_FR > 5) and (p_RL > 5) and (p_RR > 5):
            end = 1
        elif PedalPos == 0 and PW_Counter_Off > 0.5 and p_MC < 5 and v_Max - v_Min < 0.2:
            end = 1
        else:
            end = 0
    
        ends.append(end)
        
    df['TCS_End'] = ends

    throttle_list = []
    for indexs in pw_indexs:
        i_start, i_end = indexs
        df_range = df.iloc[i_start:i_end]
        if len(df_range[df_range['TCS_End'] == 1]) > 0:
            # print(indexs, len(df_range[df_range['TCS_End'] == 1]))
            i_end_updated = df_range[df_range['TCS_End'] == 1].index[0]
            if i_end_updated - i_start > 2 / 0.005:
                throttle_list.append((i_start, i_end_updated))

    # print(f'throttle_list: {throttle_list}')

    return df, throttle_list


# ## TCS_AxSum

# In[46]:


def TCS_AxSum(df_all, throttle_list, sampling):
    data = []
    
    for indexs in throttle_list:
        i_start, i_end = indexs
        df = df_all.iloc[i_start:i_end]
        
        best_loss = 10000
        best_gain = None
        best_offset = None
    
        for o in range(-20, 20, 5):
            offset = o / 100
            
            for g in range(-100, 100, 20):
                gain = 1 + g / 1000
                loss = 0
                
                for n, i in enumerate(df.index):
                    ax = df.loc[i, 'ax']
                    v_FL = df.loc[i, 'v_FL']
                    v_FR = df.loc[i, 'v_FR']
                    v_RL = df.loc[i, 'v_RL']
                    v_RR = df.loc[i, 'v_RR']
                
                    if n == 0:
                        vx = (v_FL + v_FR + v_RL + v_RR) / 4
                    else:
                        vx += (ax + offset) * sampling * gain      
                
                loss = abs((v_FL + v_FR + v_RL + v_RR) / 4 - vx)
        
                if best_loss > loss:
                    best_loss = loss
                    best_gain = gain
                    best_offset = offset
    
                # print(f'{best_offset}, {best_gain}, {loss}')
        
        sum_ax = []
        
        for n, i in enumerate(df.index):
            ax = df.loc[i, 'ax']
            v_FL = df.loc[i, 'v_FL']
            v_FR = df.loc[i, 'v_FR']
            v_RL = df.loc[i, 'v_RL']
            v_RR = df.loc[i, 'v_RR']
        
            if n == 0:
                vx = (v_FL + v_FR + v_RL + v_RR) / 4
            else:
                vx += (ax + best_offset) * sampling * best_gain

            loss = abs((v_FL + v_FR + v_RL + v_RR) / 4 - vx)
        
            # vx = max(vx, -0.1)
            sum_ax.append(vx)

        # print(f'best: {best_offset}, {best_gain}, {loss}')
        
        # df['sum_ax'] = sum_ax
        # df['vRef_sum_ax'] = sum_ax
        df.loc[:, 'vRef_sum_ax_TCS'] = sum_ax
        df.loc[:, 'vRef_sum_ax'] = sum_ax
        df_all.update(df)

    return df_all


# In[47]:


def Check_Sigs(df, sigs):
    exist = True
    
    for sig in sigs:
        if sig not in df.columns:
            exist = False
            break
            
    return exist


# # vRef

# In[48]:


def vRef(File_in, File_out, sampling=0.005, override=True, mode='ABS_TCS'):
    data = []
    
    df = pd.read_table(File_in, sep=',')
    
    if override == True or 'vRef_sum_ax' not in df.columns: 
        Sig_Exist = Check_Sigs(df, ['BlsAsw', 'p_MC_Model', 'v_FL', 'v_FR', 'v_RL', 'v_RR', 'MotPedalPosDriver', 'p_Model_FL', 'p_Model_FR', 'p_Model_RL', 'p_Model_RR', 'ax'])
    
        if Sig_Exist:
            df['vRef_sum_ax_ABS'] = 10000.
            df['vRef_sum_ax_TCS'] = 10000.
            df['vRef_sum_ax'] = (df['v_FL'] + df['v_FR'] + df['v_RL'] + df['v_RR']) / 4

            if mode == 'ABS_TCS' or mode == 'ABS':
                df, braking_list = ABS_Braking_List(df, sampling)
                df = ABS_AxSum(df, braking_list, sampling)
                # data += data_

            if mode == 'ABS_TCS' or mode == 'TCS':
                df, throttle_list = TCS_Throttle_List(df, sampling)
                # print(throttle_list)
                df = TCS_AxSum(df, throttle_list, sampling)
                # data += data_
        
            df['vRef_sum_ax'] = df['vRef_sum_ax'].rolling(window=20, center=True).mean()
            
            df.to_csv(File_out, index=False)

            if df['vRef_sum_ax_ABS'].min() == 10000. and df['vRef_sum_ax_TCS'].min() == 10000.:
                File_out = None
        else:
            File_out = None

    return File_out


# # Run

# In[49]:


# DIR = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\22MY_3A0A_CR-V\\'
# CSV = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200114_0025.csv'
# CSV_out = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200114_0025_out.csv'
# CSV = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200114_0001.csv'
# CSV_out = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200114_0001_out.csv'
# CSV = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200115_0017.csv'
# CSV_out = f'{DIR}Data\\90_PET_Data\\MS_2WD\\20200114_0220_bl7_TCS\\20200115_0017_out.csv'

# DIR = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\22MY_3A0A_CR-V\\Data\\90_PET_Data\\MS_4WD\\20200116_0211_bl7_TCS\\'
# # CSV = f'{DIR}20200206_0036.csv'
# # CSV_out = f'{DIR}20200206_0036_out.csv'
# CSV = f'{DIR}20200206_0038.csv'
# CSV_out = f'{DIR}20200206_0038_out.csv'

# DIR = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\3BVT\\Test\\'
# # CSV = f'{DIR}20200206_0036.csv'
# # CSV_out = f'{DIR}20200206_0036_out.csv'
# CSV = f'{DIR}20240213_0001.csv'
# CSV_out = f'{DIR}20240213_0001_out.csv'

# Out = vRef(CSV, CSV_out, mode='TCS')
# print(Out)


# In[ ]:




