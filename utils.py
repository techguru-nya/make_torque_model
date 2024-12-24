#!/usr/bin/env python
# coding: utf-8

# # Import

# In[2]:


import os
import sys
import datetime
import time
import subprocess

import glob

import shutil

import pandas as pd
import numpy
import numpy as np

import datetime

import zipfile

import matplotlib.pyplot as plt

import math

from scipy import integrate
from scipy import signal

import shutil

# import Read_DCM_5

# SAMPLING = 0.005


# # Convert ZIP(D97-->CSV)

# ## ReadComment

# In[3]:


def ReadComment(File):
    Comment = ''
    
    try:
        try:
            with zipfile.ZipFile(File) as zf:
                lst = zf.namelist()

                text = None

                for name in lst:
                    if ".par" not in name and ".TXT" in name:
                        data = zf.read(name)
                        Comment = OutputText(str(data))

        except BadZipFile:
            1
    except NameError:
         Comment = ''

    return Comment


def OutputText(A):
    B = A.split("\\r\\n")
    i = 0
    C = ""
    for b in B:
        if i < 8:
            C += b + ", "
            i += 1

    D = C.replace("b'", "")

    return D


# In[4]:


def RemoveTemp(file):
    is_file = os.path.isfile(file)
    
    if is_file:
        os.remove(file)


def MakePLT(Path0):
    Path1 = Path0 + "Temp.PLT"

    Text = ""
    for S in d_SIGNAL_PLT:
        Text += S + " key="+ d_SIGNAL_PLT[S] + "\n"

    f = open(Path1, "w")
    f.write(Text)
    f.close()

    return Path1


# ## DataTreatment

# In[5]:


def DataTreatment(File, d_Plt, Sampling):
    Csv = None
    NotConvert = False
    
    Folder = os.path.dirname(File) + '/'
    
    root, ext = os.path.splitext(File)
    if (ext == ".ZIP" or ext == ".zip") and "ApplContainer" not in File:
        d97 = UnPackD97(File)

        if d97 != None:
            Plt_new, d_Plt_new, NotAll = MakePLTFromD97(d97, d_Plt)
            
            if Plt_new != None:
                Csv = RunBat(d97, Plt_new, d_Plt_new, Sampling)
                
                dirname, basename = os.path.split(Csv)
                Csv_ = Folder + basename
                os.replace(Csv, Csv_)
            else:
                Csv = None
            
            if Csv == None or NotAll == True:
                NotConvert = True
                
        try:
            Remove_w_ExistFile(d97)
        except TypeError:
            print('TypeError:', File, d97)
            
    return Csv, NotConvert


def UnPackD97(file):
    file_d97 = None
    file_in_zip = ''
    
    dirname, basename = os.path.split(file)
    dirname_ = dirname + '/'
    
    try:
        try:
            with zipfile.ZipFile(file) as zf:
                lst = zf.namelist()

                for file_in_zip in lst:
                    root, ext = os.path.splitext(file_in_zip)
                    
                    if ext == ".D97" or ext == ".d97":  
                        # shutil.unpack_archive(file, dirname_, format='zip')
                        
                        with zipfile.ZipFile(file) as existing_zip:
                            existing_zip.extract(file_in_zip, dirname_)
                            
                        file_d97 = file_in_zip
                        break
                        
        except BadZipFile:
            file_d97 = None
    except NameError:
        if file_in_zip != '':
            Remove_w_ExistFile(Folder + file_in_zip)
            
        file_d97 = None
    
    if file_d97 != None:
        file_out = UnPackD97__Change_FileName(dirname_, file, file_d97)
    else:
        file_out = None
    
    return file_out


def UnPackD97__Change_FileName(Folder, ZIP, D97):    
    # path = Folder + D97
    root, ext = os.path.splitext(ZIP)
    path_new = root + '.D97'
    
    dirname, basename = os.path.split(ZIP)
    path_base = dirname + '/' + D97
    # path_new = Folder + file_name
    
    # print(path_base, path_new)
    # if path_new != path_base:
    #     if os.path.exists(path_new) == True:
    #         os.remove(path_new)
            
    os.rename(path_base, path_new)
    
    return path_new   


def RunBat(file_, Plt, d_Plt, Sampling):
    dirname, basename = os.path.split(file_)
    file = basename
    
    if ".D97" in file:
        CSV = file.replace(".D97", ".CSV")
    elif ".d97" in file:
        CSV = file.replace(".d97", ".CSV")

    # Bat = dirname + "ChangeFormat.bat"
    File0 = dirname + '/' + file
    File1 = dirname + '/' + "1__" + file
    File2 = dirname + '/' + "2__" + file
    File3 = dirname + '/' + "3__" + CSV

    # Text = "MDFDSET3c ifn=" + File0 + ";pltfn=" + Plt + " ofn=" + File1 + "\n"
    Command = "MDFDSET6c ifn=" + File0 + ";pltfn=" + Plt + " ofn=" + File1 + "\n"
    subprocess.call(Command, shell=True)
    
    # Text = "MDFMDL6c ifn=" + File0 + " ofn=" + File1 + " INCLUDE_SG=" + Plt + "\n" 
    
    Command = "MDFMDL6c ifn=" + File1 + " ofn=" + File2 + " tc=" + str(Sampling) + "\n"
    subprocess.call(Command, shell=True)
    
    Command = "SDTM3c ifn=" + File2 + " ofn=" + File3
    subprocess.call(Command, shell=True)
    
#     f = open(Bat, "w")
#     f.write(Text)
#     f.close()

#     res = subprocess.run([Bat], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    Remove_w_ExistFile(File0)
    Remove_w_ExistFile(File1)
    # Remove_w_ExistFile(Bat)

    try:        
        FileOut = ModifyCSV(File3, d_Plt)
        Remove_w_ExistFile(File2)
        Remove_w_ExistFile(File3)
        Remove_w_ExistFile(Plt)
        
    except FileNotFoundError:
        # dirname, basename = os.path.split(file)
        root, ext = os.path.splitext(file_)
        Plt_ = root + '.PLT'
        # Bat_ = root + '.bat'
        
        os.rename(Plt, Plt_)
        
        root, ext = os.path.splitext(File0)
        File0_ = root + '_.D97'
        File3_ = root + '.csv'
        
        # DFMDL6c ifn=c:/TSDE_Workarea/ktt2yk/Work/CarSim/SIM_ABS_Ice/ABS_Ice_N_Spike_base_.D97 t0=1 t1=23 ofn=c:/TSDE_Workarea/ktt2yk/Work/CarSim/SIM_ABS_Ice/1__ABS_Ice_N_Spike_base.D97
        Command = "MDFMDL6c ifn=" + File0 + " t0=0 t1=30" + " ofn=" + File0_ + "\n"
        subprocess.call(Command, shell=True)
        
        Command = "MDFDSET3c ifn=" + File0_ + ";pltfn=" + Plt_ + " ofn=" + File1 + "\n"
        subprocess.call(Command, shell=True)
        
        Command = "MDFMDL6c ifn=" + File1 + " ofn=" + File2 + " tc=" + str(Sampling) + "\n"
        subprocess.call(Command, shell=True)
        
        Command = "SDTM3c ifn=" + File2 + " ofn=" + File3_
        subprocess.call(Command, shell=True)
        
        # f = open(Bat_, "w")
        # f.write(Text_)
        # f.close()
        
        print("FileNotFoundError", Plt_)
        FileOut = None

    return FileOut


def ChangePath(Folder0, File0):
    FILE1 = File0.split("/")
    File = Folder0 + FILE1[-1]
    Folder = Folder0 + FILE1[-2]
    
    return File, Folder, FILE1[-1]


def ModifyCSV(File, d_Plt):
    for i, S in enumerate(d_Plt):
        if i == 0:
            Text_PLT = "TIME" + "," + d_Plt[S]
        else:
            Text_PLT += "," + d_Plt[S]

    with open(File) as f:
        Text_CSV = f.read()

    Text = Text_PLT + "\n" + Text_CSV
    Text = Text.replace(",", "\t")

    f = open(File, "w")
    f.write(Text)
    f.close()
    
    df = pd.read_table(File, sep="\t", index_col=0, skiprows=[1])
    
    dirname, basename = os.path.split(File)
    File2 = dirname + '/' + basename.replace("3__", "")
    # File2 = File2.replace(".CSV", ".csv")
    
    basename_without_ext = os.path.splitext(os.path.basename(File2))[0]
    dirname, basename = os.path.split(File2)
    # now = datetime.datetime.now()
    # FileOut = dirname + '\\' + basename_without_ext + '_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
    FileOut = dirname + '/' + basename_without_ext + '.csv'
    df.to_csv(FileOut)

    # ExistFile(FileOut)

    return FileOut


def Remove_w_ExistFile(PathFile):
    if os.path.exists(PathFile) == True:
        os.remove(PathFile)


# ##  MakePLTFromD97

# In[6]:


def MakePLTFromD97(File_D97, d_Plt):
    l_Signals_new = []
    d_Plt_new = {}
    
    dirname, basename = os.path.split(File_D97)
    dirname_ = dirname + '/'
    
    D97 = File_D97

    if os.path.exists(D97) == True:
        l_Signals_D97 = ReadD97(D97)
        l_Signals_PLT = list(d_Plt.keys())
        
        for T in l_Signals_PLT:
            if T in l_Signals_D97:
                l_Signals_new.append(T)
            else:
                Error = "Error: " + T + " is nothing."
                # print(Error)

        d_Plt2 = {}
        for S in d_Plt:
            if d_Plt[S] in d_Plt2:
                d_Plt2[d_Plt[S]] = d_Plt2[d_Plt[S]] + [S]
            else:
                d_Plt2[d_Plt[S]] = [S]

        # print(1, d_Plt2)
        NotAll = False
        for S in d_Plt2:
            Found = False
            
            for S1 in d_Plt2[S]:
                if S1 in l_Signals_new:
                    d_Plt_new[S1] = S
                    Found = True
                    break
            
            if Found == False:
                NotAll = True
                    
        # print(2, d_Plt_new)

        Text = ""
        for T in d_Plt_new:
            Text += T + "\n"
        
        # print(3, Text)
                
        Plt_new = dirname_ + 'Temp.PLT'
        
        if Text != "":
            f = open(Plt_new, 'w')
            f.write(Text)
            f.close()
        else:
            Plt_new = None
            d_Plt_new = None
            NotAll = False
    else:
        Plt_new = None    
        d_Plt_new = None
        NotAll = False

    return Plt_new, d_Plt_new, NotAll


def ReadD97(Path):
    Out = []

    #f=open(Path, 'r', encoding="utf_8")
    f = open(Path, 'rb')

    i = 2
    while True:
        line_b = f.readline()
        line = str(line_b)

        # if "[SIGNAL0]" not in line and "[SIGNAL" in line:
        if "[SIGNAL" in line:
            i = 0

        if i == 1 and "NAME=" in line:
            T = line.replace("NAME=", "")
            T = T.replace("\n", "")
            T = T.replace("b'", "")
            T = T.replace("'", "")
            T = T.replace("\\", "*")
            T = T.replace("*r*n", "")

            Out.append(T)

        if "[DATA]" in line:
            break

        i += 1

    f.close()

    return Out


# ## MakeTraceList

# In[7]:


def MakeTraceList(l_Folder, l_Ext, l_Ext_wo):
    print(l_Folder)
    
    l_Traces = []
    
    for Folder in l_Folder:
        for current, subfolders, subfiles in os.walk(Folder):
            for file in subfiles:
                if "ApplContainer" not in file:
                    for Ext_ in l_Ext:
                        if Ext_ in file: 
                            Trace = current + '/'+ file
                            # Trace = current + file
                            l_Traces.append(Trace)
                            break
    
    l_Traces_ = []
    for T in l_Traces:
        Delete = False
        for Ext in l_Ext_wo:
            if Ext in T:
                Delete = True
                break
                
        if Delete == False:
            l_Traces_.append(T)

    return l_Traces_


def CopyMeasurement(df, Folder):
    df1 = df.dropna(subset=["File"])
    L_Measurement = list(df1["File"])

    L_Measurement_New = []

    i = 0
    for Path in L_Measurement:
        FileName = os.path.basename(Path)

        Path1 = Folder + FileName
        if Path1 not in L_Measurement_New:
            L_Measurement_New.append(Path1)
        else:
            FileName1 = FileName.replace(".zip", "")
            FileName1 = FileName1.replace(".ZIP", "")
            Folder_new = Folder + FileName1 + "_" + str(i)
            os.mkdir(Folder_new)
            Path1 = Folder_new + "/" + FileName
            L_Measurement_New.append(Path1)
        i += 1

    i = 0
    for M in L_Measurement:
        try:
            shutil.copy(M, L_Measurement_New[i])
        except FileNotFoundError:
            0
        i += 1


# ## Select_Signal
# - D97ファイルをCSV変換するためのPLT作成で使用する信号
# - PLTから計測信号を設定する。

# In[8]:


def Select_Signal(File):
    d_Signal = {}
    
    f = open(File, 'r', encoding="ascii")
    l_line_plt = f.readlines()

    for line in l_line_plt:
        line = line.replace("\n", "")
        l_line = line.split(" ")

        if l_line[0] != "" and "~" not in l_line[0] and "//" not in l_line[0] and "+" not in l_line[0] and "*" not in l_line[0]:
            d_Signal[l_line[0]] = GetKey(l_line)

    f.close()

    return d_Signal


def GetKey(l_in):
    Out = l_in[0]
    
    for Text in l_in:
        l_Text = Text.split("=")
        
        if l_Text[0] == "key":
            Out = l_Text[1]
            
    return Out


# ## SaveData, ReadData

# In[9]:


def SaveData(Data1, Data2, File):
    # basename_without_ext = os.path.splitext(os.path.basename(File))[0]
    # dirname, basename = os.path.split(File)
    root, ext = os.path.splitext(File)
    now = datetime.datetime.now()
    File_new = root + '___' + now.strftime('%Y%m%d_%H%M%S') + ext
    
    print('Save:', File_new)
    pd.to_pickle((Data1, Data2), File_new)
    
    return File_new


def ReadData(File):
    # basename_without_ext = os.path.splitext(os.path.basename(File))[0]
    dirname, basename = os.path.split(File)
    root, ext = os.path.splitext(File)
    # now = datetime.datetime.now()
    
    l_Files = os.listdir(dirname)
    l_date = []
    
    for F in l_Files:
        root1, ext1 = os.path.splitext(dirname + '/' + F)
        dirname1, basename1 = os.path.split(dirname + '/' + F)
        
        if ext == ext1:
            if '___' in basename1:
                l_basename1 = basename1.split('___')
                basename0, ext0 = os.path.splitext(basename)

                if basename0 == l_basename1[0]:                
                    date = l_basename1[-1]
                    date1, date1_ext = os.path.splitext(date)
                    l_date.append(date1)
                    # print(2, date)
                
    l_date.sort()
    print(l_date)
    
    File_new = root + '___' + l_date[-1] + ext
    
    print('Read:', File_new)
    Data1, Data2 = pd.read_pickle(File_new)
    
    return Data1, Data2


# ## SAVE_ZIP_to_CSV

# In[10]:


def SAVE_ZIP_to_CSV(d_Plt, Folder_In, Sampling, Override):
    d_Csvs = {}
    l_None = []
    
    l_Traces = MakeTraceList(Folder_In, ['.ZIP', '.zip'], [])
    
    # for Zip in l_Traces:
    for i, Zip in enumerate(l_Traces):
        Size = os.path.getsize(Zip)
        print(i + 1, '/', len(l_Traces), ';', Zip, Size)
        
        dirname, basename = os.path.split(Zip)
        Csv_ = dirname + '\\' + basename 
        Csv_ = Csv_.replace('.ZIP', '.csv')
        
        if os.path.exists(Csv_) == False or Override == True:
            Csv, NotConvert = DataTreatment(Zip, d_Plt, Sampling)
        else:
            Csv = Csv_
            NotConvert = False
            
        Text = ReadComment(Zip)
        
        if Csv != None:
            d_Csvs[Zip] = (Text, Csv, (d_Plt, Sampling, Size))
        
        if NotConvert == True:
            l_None.append(Zip)
    
    return d_Csvs, l_None


# ## SAVE_CSV_to_D97_w_ZIP

# In[11]:


def SAVE_CSV_to_D97_w_ZIP(df, File):
    root, ext = os.path.splitext(File)
    dirname = os.path.dirname(File)
    
    CSV = root + '_mod.csv'
    ZIP = root + '_mod.zip'
    D97 = root + '_mod.d97'
    D97_ = os.path.basename(D97)
    BAT = dirname + '/' + '_.bat'
    
    df.to_csv(CSV, index=False)
    
    Command = "SDTM3c ifn=" + CSV + " ofn=" + D97
    subprocess.call(Command, shell=True)
    
#     f = open(BAT, "w")
#     f.write(Text)
#     f.close()

#     res = subprocess.run([BAT], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
 
    compFile = zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED)
    compFile.write(D97, arcname=D97_)
    compFile.close()
    
    # Remove_w_ExistFile(D97)
    Remove_w_ExistFile(CSV)
    # Remove_w_ExistFile(BAT)
    
    return ZIP


# ## SAVE_CSV_to_ZIP

# In[12]:


def SAVE_CSV_to_ZIP(File, remove_csv=False):
    if os.path.isfile(File):
        file_paths = [File]
        root, ext = os.path.splitext(File)
        output_zip = f'{root}.zip'
        # zip_files(file_paths, output_zip)
        zip_files_with_compression(file_paths, output_zip)
    
        if remove_csv == True:
            os.remove(File)
            
        Out = True
    else:
        Out = False

    return Out


def zip_files(file_paths, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in file_paths:
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"File {file} does not exist")


def zip_files_with_compression(file_paths, output_zip, compression_level=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(output_zip, 'w', compression=compression_level) as zipf:
        for file in file_paths:
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"File {file} does not exist")


# ## Unpack_ZIP_to_CSV

# In[13]:


def Unpack_ZIP_to_CSV(File, remove_zip=False):
    if os.path.isfile(File):
        zip_path = File  # Zipファイルのパス
        output_path = os.path.dirname(File)
        extracted_files = extract_all_csv_from_zip(zip_path, output_path)

        if remove_zip == True:
            os.remove(File)
    else:
        extracted_files = []
        
    return extracted_files


def extract_all_csv_from_zip(zip_path, output_path):
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Zipファイル内のすべてのファイル名を取得
        file_names = zipf.namelist()
        # CSVファイルのみを抽出
        csv_files = [file for file in file_names if file.endswith('.csv')]
        
        for csv_file in csv_files:
            zipf.extract(csv_file, output_path)
            print(f"Extracted {csv_file} to {output_path}")
            extracted_files.append(f'{output_path}/{csv_file}')

    return extracted_files


# ## Run 

# In[14]:


# PLT_Convert = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Common\\ABS\\HONDA_32RA_ABS.PLT' 
# PATH_Search = ['c:\\TSDE_Workarea\\ktt2yk\\Work\\Common\\ABS']
# SAMPLING = 0.005 

# d_SIGNAL_PLT = Select_Signal(PLT_Convert)
# d_CSVs, l_Not_Convert = SAVE_ZIP_to_CSV(d_SIGNAL_PLT, PATH_Search, SAMPLING, True)           


# In[15]:


# l_csvs = MakeTraceList(['c:\\TSDE_Workarea\\ktt2yk\\Work\\Test\\'], ['.csv'], [])

# for csv in l_csvs:
#     SAVE_CSV_to_ZIP(csv, remove_csv=True)

# l_zips = MakeTraceList(['c:\\TSDE_Workarea\\ktt2yk\\Work\\Test\\'], ['.zip'], [])

# for zip in l_zips:
#     outs = Unpack_ZIP_to_CSV(zip, remove_zip=True)
#     print(outs)


# # Reampling_CSV

# In[16]:


def Resampling_CSV(File, sampling_time='5ms'):
    # 例: 実際のデータフレームを読み込む
    df = pd.read_csv(File)
    
    # 'seconds'列が存在し、秒単位のデータとして扱う
    df['timestamp'] = pd.to_datetime(df['TIME'], unit='s', origin='unix')
    
    # データフレームのインデックスを'timestamp'に設定
    df.set_index('timestamp', inplace=True)
    
    # 5ミリ秒間隔にリサンプリング
    # df_resampled = df.resample('5ms').mean()
    df_resampled = df.resample(sampling_time).mean()
    
    # print(df_resampled.head(10))

    root, ext = os.path.splitext(File)
    resampled_file = f'{root}_{sampling_time}.csv'
    df_resampled.to_csv(resampled_file, index=False)

    return resampled_file


# In[17]:


# file = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Common\\20220119_0004_2138_3A0A_ABS_FHEV_KA_4WD.csv'
# Sampling(file, sampling_time='5ms')


# # READ COMMENT in TRACE

# ## COMMENT_LIST

# In[18]:


def COMMENT_LIST(l_Path):
    l_Trace = MakeTraceList(l_Path, ['.zip', '.ZIP'], [])

    d_Comment = {}

    for Tra in l_Trace:
        Com = ReadComment(Tra)
        d_Comment[Tra] = Com
        
    return d_Comment


# ## COMMENT_SEARCH

# In[19]:


def COMMENT_SEARCH(d_Comment, l_Key, l_Key_Remove):
    d_Searched = {}
    
    for Tra in d_Comment:
        Com = d_Comment[Tra]
        
        Counter = 0
        for Key in l_Key:
            if Key.lower() in Com.lower():
                Counter += 1
        
        if Counter == len(l_Key):
            d_Searched[Tra] = Com
    
    l_Remove = []
    for i, Tra in enumerate(d_Searched):
        Com = d_Searched[Tra]

        for Key in l_Key_Remove:
            if Key.lower() in Com.lower():
                l_Remove.append(Tra)
                break

    for Tra in l_Remove:
        d_Searched.pop(Tra)     
    
    return d_Searched


# ## COPY_FILE

# In[33]:


def COPY_FILE(l_File, dst_dir, src_dir_rmv_key):
    l_Out = []
    
    for File in l_File:
        src_file = File.replace(src_dir_rmv_key, '')
        src_dir = os.path.dirname(src_file)
        dst_path = dst_dir + src_dir + '\\'
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # print(File, dst_path)
        
        basename = os.path.basename(File)
        Out = dst_path + basename
        # print(File, Out)

        shutil.copy2(File, Out)
        
        l_Out.append(Out)
        
    return l_Out


# ## RUN

# In[ ]:


# l_PATH = ['d:\\Projects\\22MY_3A0A_CR-V\\']
# d_COM = COMMENT_LIST(l_PATH)

# import pickle

# with open('3A0A_Data_List.pkl', 'wb') as file:
#     pickle.dump(d_COM, file)

# with open('3A0A_Data_List.pkl', 'rb') as file:
#     d_COM = pickle.load(file)

# d_SEARCH_Snow_Stdless = COMMENT_SEARCH(d_COM, ['Snow'], [])
# d_SEARCH_Ice_Stdless = COMMENT_SEARCH(d_COM, ['Ice'], [])

# print(f'Snow:{len(list(d_SEARCH_Snow_Stdless.keys()))}, Ice:{len(list(d_SEARCH_Ice_Stdless.keys()))}')

# destination_directory = 'c:\\TSDE_Workarea\\ktt2yk\\Projects\\'
# source_directory_remove_key = 'd:\\Projects\\'
# l_OUT = COPY_FILE(list(d_SEARCH_Ice_Stdless), destination_directory, source_directory_remove_key)

# l_OUT = COPY_FILE(list(d_SEARCH_Snow_Stdless), destination_directory, source_directory_remove_key)


# # Modify Signal

# ## Modify_Signal_CSSIM_INPUT

# In[13]:


def Modify_Signal_CSSIM_INPUT(l_Traces, File):
    l_Out = []
    
    d_Parameter = Read_DCM(File)
    Abrollumfang_VA = d_Parameter['Abrollumfang_VA'][2][0]
    Abrollumfang_HA = d_Parameter['Abrollumfang_HA'][2][0]
    
    for Csv in l_Traces:
        print(Csv)
        df = pd.read_table(Csv, sep=",", index_col=None)
        
        # df['nMotNET_TRC'] = df['nMotNET_TRC'] / (2 * 3.14 / 60)        
        df['Ays'] = df['Ays'] * (-1)
        df['Yrs'] = df['Yrs'] * (-1)
        df['v_FL'] = df['v_FL'] * 3.6 * ((1000 / 3600) / Abrollumfang_VA * 60)
        df['v_FR'] = df['v_FR'] * 3.6 * ((1000 / 3600) / Abrollumfang_VA * 60)
        df['v_RL'] = df['v_RL'] * 3.6 * ((1000 / 3600) / Abrollumfang_HA * 60)
        df['v_RR'] = df['v_RR'] * 3.6 * ((1000 / 3600) / Abrollumfang_HA * 60)
        df['SasInCor'] = df['SasInCor'] * (-1) * 180 / 3.14
        # df['p_MC_Model'] = df['p_MC_Model'] * 10
        
        # if 'nMotNET_SMU' in df.columns:
        #     df['nMotNET_SMU'] = df['nMotNET_SMU'] / (2 * 3.14 / 60)
        
        root, ext = os.path.splitext(Csv)
        Csv_ = root + '_mod' + ext
        df.to_csv(Csv_, header=False, index=False)
        
        l_Out.append(Csv_)
        
    return l_Out


# In[14]:


def Modify_Signal_CSSIM_INPUT_pMC_10(l_Traces, File):
    l_Out = []
    
    d_Parameter = Read_DCM(File)
    Abrollumfang_VA = d_Parameter['Abrollumfang_VA'][2][0]
    Abrollumfang_HA = d_Parameter['Abrollumfang_HA'][2][0]
    PT_DT_DiffRatio_Axle = d_Parameter['PT_DT_DiffRatio_Axle'][2][0]
    PT_DT_DiffRatio_Axle_2 = d_Parameter['PT_DT_DiffRatio_Axle_2'][2][0]
    
    for Csv in l_Traces:
        print(Csv)
        df = pd.read_table(Csv, sep=",", index_col=None)
        
        omega_FA = (df['v_FL'] + df['v_FR']) * 0.5 / (Abrollumfang_VA / 2 / math.pi)
        omega_RA = (df['v_RL'] + df['v_RR']) * 0.5 / (Abrollumfang_HA / 2 / math.pi)
        
        df['nMotNET_TRC'] = 0.995 * omega_RA * PT_DT_DiffRatio_Axle_2 / (2 * math.pi / 60.0)
        df['nMotNET_SMU'] = 0.955 * omega_FA * PT_DT_DiffRatio_Axle / (2 * math.pi / 60.0)
        
        # df['nMotNET_TRC'] = df['nMotNET_TRC'] / (2 * 3.14 / 60)        
        df['Ays'] = df['Ays'] * (-1)
        df['Yrs'] = df['Yrs'] * (-1)
        df['v_FL'] = df['v_FL'] * 3.6 * ((1000 / 3600) / Abrollumfang_VA * 60)
        df['v_FR'] = df['v_FR'] * 3.6 * ((1000 / 3600) / Abrollumfang_VA * 60)
        df['v_RL'] = df['v_RL'] * 3.6 * ((1000 / 3600) / Abrollumfang_HA * 60)
        df['v_RR'] = df['v_RR'] * 3.6 * ((1000 / 3600) / Abrollumfang_HA * 60)
        df['SasInCor'] = df['SasInCor'] * (-1) * 180 / 3.14
        df['p_MC_Model'] = df['p_MC_Model'] / 10
        
        # if 'nMotNET_SMU' in df.columns:
        #     df['nMotNET_SMU'] = df['nMotNET_SMU'] / (2 * 3.14 / 60)
        
        root, ext = os.path.splitext(Csv)
        Csv_ = root + '_mod' + ext
        df.to_csv(Csv_, header=False, index=False)
        
        l_Out.append(Csv_)
        
    return l_Out


# ## Modify_Signal_Simout

# In[15]:


def Modify_Signal_Simout(l_Traces, File):
    l_Out = []
    
    d_Parameter = Read_DCM(File)
    Cp_FA = d_Parameter['Abrollumfang_VA'][2][0]
    Cp_RA = d_Parameter['Abrollumfang_HA'][2][0]
    
    for Csv in l_Traces:
        print(Csv)
        df = pd.read_table(Csv, sep=",", index_col=None)
        
        # df['nMotNET_TRC'] = df['nMotNET_TRC'] / (2 * 3.14 / 60)        
        df['BRK_TRQ_FL'] = df['BRK_TRQ_FL'] * Cp_FA
        df['BRK_TRQ_FR'] = df['BRK_TRQ_FR'] * Cp_FA
        df['BRK_TRQ_RL'] = df['BRK_TRQ_RL'] * Cp_RA
        df['BRK_TRQ_RR'] = df['BRK_TRQ_RR'] * Cp_RA
        
        root, ext = os.path.splitext(Csv)
        Csv_ = root + '_mod' + ext
        df.to_csv(Csv_, header=True, index=False)
        
        l_Out.append(Csv_)
        
    return l_Out


# ## Modify_Signal

# In[16]:


def Modify_Signal(l_Traces, d_Sig):
    l_Out = []
    
    for Csv in l_Traces:
        print(Csv)
        df = pd.read_table(Csv, sep=",", index_col=None)
        
        for S in d_Sig:
            if S in df.columns:
                df[S] = df[S] * d_Sig[S]
        
        root, ext = os.path.splitext(Csv)
        Csv_ = root + '_sig' + ext
        df.to_csv(Csv_, header=True, index=False)
        
        l_Out.append(Csv_)
        
    return l_Out


# In[17]:


def DIFF(data, n, Sampling):
    data_ = data.diff() / Sampling
    
    if n != 0:
        data_ = data_.rolling(n, center=False).mean()
    
    data_ = data_.fillna(0)
    
    return data_


def INTEG(x, y):
    integ = integrate.cumtrapz(y, x)
    integ_ = [0] + list(integ)
    
    return integ_
    

# data_ = LOWPASS(list(data_), (0.5, 5, 1, 2))
def LOWPASS(x, _):
    samplerate = 1 / 0.005                                   #波形のサンプリングレート
    
    # fp = 0.5       #通過域端周波数[Hz]
    # fs = 5       #阻止域端周波数[Hz]
    # gpass = 1       #通過域端最大損失[dB]
    # gstop = 2      #阻止域端最小損失[dB]
    fp, fs, gpass, gstop = _ 

    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")  
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    
    return y                                      #フィルタ後の信号を返す


# ## Run

# In[18]:


# OUT_PATH = 'c:/TSDE_Workarea/ktt2yk/Work/Traces/Honda-e/Winter_Fix/SIM_Vehicle/'
# DCM = 'c:/TSDE_Workarea/ktt2yk/Work/DCM/Honda-e_20230228/Honda-e/Complete_ESP10CB_VarCode_1_Honda_e.dcm'

# l_TRACE = MakeTraceList([OUT_PATH], ['.CSV', '.csv'], ['_mod.csv'])

# l_TRACE = Modify_Signal_CSSIM_INPUT(l_TRACE, DCM)
# l_TRACE = Modify_Signal_Simout(l_TRACE, DCM)


# # DATAFRAME

# ## CONCAT_DATA

# In[19]:


def CONCAT_DATA(l_DATA):
    n = 0
    
    for i, _ in enumerate(l_DATA):
        df_ = _.copy()
        n1 = n + len(df_)
        l_Index = list(np.arange(n, n1))
        # df_['Index'] = l_Index
        df_.loc[:, 'Index'] = l_Index
        df_ = df_.set_index('Index')

        if i == 0:
            df = df_
        else:
            df = pd.concat([df, df_])

        n = n1 + 1

    return df


# ## CONCAT_DATA_w_Transition

# In[1]:


def CONCAT_DATA_w_Transition(l_DATA, T_Transition):
    l_df = []
    df_k1 = None
    TIME = 0
    
    for i, df in enumerate(l_DATA):
        df_1, df_2, TIME = Dataframe_Transition(i, TIME, df_k1, df)

        l_df.append(df_1)
        l_df.append(df_2)

        df_k1 = df

    df_concat = CONCAT_DATA(l_df)
    
    df_concat['datetime'] = pd.to_datetime(df_concat['TIME']* 1000, unit='ms')
    df_concat.set_index('datetime', inplace=True)
    df_resampling = df_concat.resample('5L').mean()
    df_resampling.interpolate(method='linear', inplace=True)

    return df_resampling


# In[2]:


def Dataframe_Transition(i, T, df1, df2):
    data = {'TIME': [T, T+5]}
    df2['TIME'] = df2['TIME'] + T + 5.005
    T_out = df2['TIME'].max()
    
    l_col = list(df2.columns)
    l_col.remove('TIME')
    
    if i == 0:
        df2_ = df2.iloc[0]
        
        for S in l_col:
            data[S] = [df2_[S], df2_[S]]
    
    else:
        df1_ = df1.iloc[df1.index.max()]
        df2_ = df2.iloc[0]
        
        for S in l_col:
            data[S] = [df1_[S], df2_[S]]
    
    df = pd.DataFrame(data)
    
    return df, df2, T_out


# # CSSIM(Matlab) Run File

# ## MATLAB_RUN_FILE_ZIP

# In[20]:


def MATLAB_RUN_FILE_ZIP(l_Trace, Mdl, Path):
    Text = ''
    Out = Path + 'Matlab_Run_w_ZIP.m'
    
    for T in l_Trace:
        dirname, basename = os.path.split(T)
        root_, ext = os.path.splitext(basename)
        root = root_.replace('_mod', '')
        File_ = root + '_simout.zip'
        File_Sim = root + '_simout.xls'
        
        Text += 'delete INPUT.csv\n'
        Text += 'delete simout.xls\n'
        Text += 'copyfile ' + basename + ' INPUT.csv\n'
        Text += "sim('" + Mdl + "')\n"
        Text += 'movefile oml_bbxxxxx.zip ' + File_ + '\n'
        Text += 'delete INPUT.csv\n'
        Text += 'copyfile simout.xls ' + File_Sim + '\n'
        Text += '\n'
        
    f = open(Out, "w")
    f.write(Text)
    f.close()
    
    return Out


# ## MATLAB_RUN_FILE_D97

# In[21]:


def MATLAB_RUN_FILE_D97(l_Trace, Mdl, Path):
    Text = ''
    Out = Path + 'Matlab_Run_w_D97.m'
    
    for T in l_Trace:
        dirname, basename = os.path.split(T)
        root_, ext = os.path.splitext(basename)
        root = root_.replace('_mod', '')
        File_d97 = root + '.d97'
        File_zip = root + '_simout.zip'
        File_Sim = root + '_simout.xls'
        
        Text += 'delete INPUT.csv\n'
        Text += 'delete simout.xls\n'
        Text += 'copyfile ' + basename + ' INPUT.csv\n'
        Text += "sim('" + Mdl + "')\n"
        Text += 'movefile oml_bbxxxxx.d97 ' + File_d97 + '\n'
        Text += "zip('" + File_zip + "' , '" + File_d97 + "')\n"
        Text += 'delete INPUT.csv\n'
        Text += 'delete ' + File_d97 + '\n'
        Text += 'copyfile simout.xls ' + File_Sim + '\n'
        Text += '\n'
        
    f = open(Out, "w")
    f.write(Text)
    f.close()
    
    return Out


# ## MATLAB_RUN_FILE_wo_D97

# In[22]:


def MATLAB_RUN_FILE_wo_D97(l_Trace, Mdl, Path):
    Text = ''
    Out = Path + 'Matlab_Run_wo_D97.m'
    
    for T in l_Trace:
        dirname, basename = os.path.split(T)
        root_, ext = os.path.splitext(basename)
        root = root_.replace('_mod', '')
        # File_d97 = root + '_OOL.d97'
        # File_zip = root + '_OOL.zip'
        File_Sim = root + '_simout.xls'
        
        Text += 'delete INPUT.csv\n'
        Text += 'delete simout.xls\n'
        Text += 'copyfile ' + basename + ' INPUT.csv\n'
        Text += "sim('" + Mdl + "')\n"
        # Text += 'movefile oml_bbxxxxx.d97 ' + File_d97 + '\n'
        # Text += "zip('" + File_zip + "' , '" + File_d97 + "')\n"
        Text += 'delete INPUT.csv\n'
        # Text += 'delete ' + File_d97 + '\n'
        Text += 'copyfile simout.xls ' + File_Sim + '\n'
        Text += '\n'
        
    f = open(Out, "w")
    f.write(Text)
    f.close()
    
    return Out


# ## Run

# In[23]:


# SEARCH_PATH = ['c:/TSDE_Workarea/ktt2yk/Work/Traces/XT1E/SIM/Winter_Fix/2023_0222__Jenkins_N245_2ChTrqLatest_wBugFix/', 'c:/TSDE_Workarea/ktt2yk/Work/Traces/XT1E/SIM/Winter_Fix/2023_0302_XT1E_Bigslip/']
# OUT_PATH = 'c:/TSDE_Workarea/ktt2yk/Work/Traces/Honda-e/Winter_Fix/SIM_Vehicle/'

# MDL = 'HM_BB86152_Var01_M_TCS_HondaE_RWD_20230315.mdl'
# MDL = 'HM_BB86153_Var01_M_TCS_XT1E_4WD_20230317_OOL.mdl'
# MDL = 'HM_BB86153_Var01_M_TCS_XT1E_4WD_20230317.mdl'

# l_TRACE = MakeTraceList(SEARCH_PATH, ['_mod.CSV', '_mod.csv'], [])

# MATLAB_RUN_FILE_ZIP(l_TRACE, MDL, OUT_PATH)
# MATLAB_RUN_FILE_D97(l_TRACE, MDL, OUT_PATH)
# MATLAB_RUN_FILE_wo_D97(l_TRACE, MDL, OUT_PATH)


# # CSSIM SIMOUT to CSV

# ## SIMOUT_to_CSV

# In[24]:


def SIMOUT_to_CSV(l_Trace, Plt):
    l_Out = []
    
    d_Signal = Select_Signal(Plt)
    
    for T in l_Trace:
        df = pd.read_excel(T, header=None)
        df_ = pd.read_excel(T, header=None)
        
        for e, Col in enumerate(df.columns):
            if e == 0:
                df_ = df_.rename(columns={e: "TIME"})
            else:
                S = list(d_Signal.keys())[e - 1]
                df_ = df_.rename(columns={e: S})
        
        root, ext = os.path.splitext(T)
        T_ = root + '.csv'
        df_.to_csv(T_, index=False)
        
        l_Out.append(T_)
    
    return l_Out


# ## Run

# In[25]:


# SEARCH_PATH_SIMOUT = ['c:/TSDE_Workarea/ktt2yk/Work/Traces/Honda-e/Winter_Fix/SIM_OOL_w_D97/', 'c:/TSDE_Workarea/ktt2yk/Work/Traces/Honda-e/Winter_Fix/SIM_OOL_wo_D97/']
# PLT_SIMOUT = 'c:/TSDE_Workarea/ktt2yk/Work/PLT/MTCS_MEDC_SIMOUT_1CH.PLT'

# l_TRACE = MakeTraceList(SEARCH_PATH_SIMOUT, ['_simout.xls'], [])
# SIMOUT_to_CSV(l_TRACE, PLT_SIMOUT)


# # PATH

# In[26]:


def CHANGE_PATH(Path):
    Path = Path.replace('c:/TSDE_Workarea/ktt2yk', '')
    Path = Path.replace('c:\\TSDE_Workarea\\ktt2yk', '')
    Path = Path.replace('\\', '/')
    
    return Path


# # PLOT to COMPARE 

# ## READ_DATA

# In[27]:


def READ_DATA(l_Trace, Signal, Time, Delay):
    l_Out = []
    
    S1, S2 = Signal
    T1, T2 = Time
    
    for S2_ in S2:
        S2__ = S2[S2_]
        
        X = []
        Y = []
        L = []

        for e, F in enumerate(l_Trace):
            Csv = l_Trace[e]
            df = pd.read_table(Csv, sep=",", index_col=None)
            df['TIME'] = df['TIME'] + Delay[e]
            # df.drop(columns=['TIME', 'C'])
            
            Con = 'TIME >= ' + str(T1) + ' and TIME <= ' + str(T2) 
            df_ = df.query(Con)
            x = df_[S1]
            y = df_[S2_]
            
            X.append(x)
            Y.append(y)
            L.append(F)

        Out = (S1, S2__, X, Y, L)
        l_Out.append(Out)
    
    return l_Out


# ## PLOT

# In[28]:


def PLOT(d_Trace, PLT, _, fig_setting, Dir):    
    d_ = Select_Signal(PLT)
    SIGNAL = ('TIME', d_)
    TIME, DELAY, LABEL = _
    
    figsize_x_, figsize_y_, dpi_ = fig_setting[0]
    left_, right_, bottom_, top_ = fig_setting[1]
    
    l_Fig = []
    
    for e, title in enumerate(d_Trace):
        l_Trace_ = d_Trace[title]

        for T in l_Trace_:
            print(T)

        if len(TIME) == 1:
            TIME_ = TIME[0]
        else:
            TIME_ = TIME[e]

        if len(DELAY) == 1:
            DELAY_ = DELAY[0]
        else:
            DELAY_ = DELAY[e]

        if len(LABEL) == 1:
            LABEL_ = LABEL[0]
        else:
            LABEL_ = LABEL[e]

        # FIG_ = FIG[e]
        # l_DATA_ = l_DATA[e]
        # root, ext = os.path.splitext(l_Trace_[0])
        # dirname, basename = os.path.split(l_Trace_[0])
        FIG_ = Dir + title + '.png'

        # basename = os.path.basename(l_Trace_[0])
        # root, ext = os.path.splitext(basename)
        # Title = root

        l_DATA = READ_DATA(l_Trace_, SIGNAL, TIME_, DELAY_)
        
        PLOT = False
        for n, _ in enumerate(l_DATA):
            S1, S2, X, Y, Label = _
            
            if len(X) >= 2:
                PLOT = True
                break
        
        if PLOT == True:
            fig = plt.figure(figsize=(figsize_x_, figsize_y_), dpi=dpi_)
            fig.subplots_adjust(left=left_, right=right_, bottom=bottom_, top=top_)

            for n, _ in enumerate(l_DATA):
                S1, S2, X, Y, Label = _

                n1 = len(l_DATA)
                ax = fig.add_subplot(n1, 1, n+1)

                x_lim0, x_lim1 = RANGE_LIMIT(X, TIME_)

                for m, x in enumerate(X):
                    y = Y[m]

                    # root, ext = os.path.splitext(Label[m])
                    # ax.plot(x, y, label=root)
                    ax.plot(x, y, label=LABEL_[m])

                    # ax.set_ylabel(S2)
                    ax.set_ylabel(S2[-14:])
                    # ax.set_xlim(TIME[0], TIME[1])
                    ax.set_xlim(x_lim0, x_lim1)

                if n < len(l_DATA) - 1:
                    ax.tick_params(labelbottom=False)

                plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 

            ax.set_xlabel(S1)
            # plt.suptitle(root, fontsize=10)
            plt.suptitle(title)
            plt.show()

            fig.savefig(FIG_)
            l_Fig.append(FIG_)
        
    return l_Fig


def RANGE_LIMIT(X, T):
    T_Min = T[0]
    T_Max = T[1]
    
    for e, x in enumerate(X):
        T_Min = max(T_Min, min(x))
        T_Max = min(T_Max, max(x))
    
    return T_Min, T_Max


# ## COMPARE LIST

# In[29]:


def COMPARE_LIST(Path, l_Key):
    d_Out = {}
    
    l_Traces = MakeTraceList(Path, l_Key, [])
    # print(l_Traces)
    
    l_basename = []
    for T in l_Traces:
        dirname, basename = os.path.split(T)
        
        basename_ = basename
        for Key in l_Key:
            basename_ = basename_.replace(Key, '')
        l_basename.append(basename_)
        
    l_basename_ = list(set(l_basename))        
    
    for basename in l_basename_:        
        l_Out1 = []
        for T in l_Traces:
            if basename in T:
                l_Out1.append(T)
        
        l_Out2 = []
        for Key in l_Key:
            for Out1 in l_Out1:
                if Key in Out1:
                    l_Out2.append(Out1)
                    l_Out1.remove(Out1)

        d_Out[basename] =l_Out2
        
    return d_Out


# In[30]:


# COMPARE_PATH_3 = ['c:/TSDE_Workarea/ktt2yk//Work/Traces/XT1E/SIM/Winter_Fix/SIM_Vehicle/', 'c:/TSDE_Workarea/ktt2yk/Work/Traces/XT1E/SIM/Winter_Fix/SIM_OOL_wo_D97_20230327/']
# OUT_PATH = 'c:/TSDE_Workarea/ktt2yk/Work/Traces/XT1E/SIM/Winter_Fix/SIM_OOL_wo_D97_20230327/'
# PLT_VIEW = 'c:/TSDE_Workarea/ktt2yk/Work/PLT/MTCS_MEDC_SIMOUT_VIEW_2CH.PLT'
# FIG_SETTING = ((10, 6, 100), (0.07, 0.90, 0.1, 0.95))

# TIME = [(1, 1000)]
# DELAY = [[0, 0]]
# # LABEL = [['Sim1', 'Sim2']]
# LABEL = [['Veh', 'Sim']]

# # d_COMPARE = COMPARE_LIST(COMPARE_PATH_1, ['_simout.csv', '_simout.csv'])
# # d_COMPARE = COMPARE_LIST(COMPARE_PATH_2, ['_simout.csv', '_mod.csv'])
# d_COMPARE = COMPARE_LIST(COMPARE_PATH_3, ['_mod.csv', '_simout.csv'])

# l_FIG = PLOT(d_COMPARE, PLT_VIEW, (TIME, DELAY, LABEL), FIG_SETTING, OUT_PATH)


# # Read DCM

# ## パラメータ検出キー

# In[2]:


PARAMETER_TYPE = ["FESTWERT", "KENNLINIE", "KENNFELD"]


# ## DCM読込み（テキストモード）

# In[3]:


def ParameterName(TEXT):
    for TYPE in PARAMETER_TYPE:
        TEXT = TEXT.replace(TYPE, "")

    TEXT = TEXT.replace("\n", "")

    P = TEXT.split(" ")

    return P[1]


def Read_DCM_w_Text(FILE):
    l_line = []

    i = 0

    try:
        f = open(FILE, 'r')

        while True:
            line = f.readline()
            l_line.append(line)

            if line == "":
                i += 1

            if i > 100:
                break

    except UnicodeDecodeError:
        try:
            f = open(FILE, 'r', encoding='shift-jis')

            while True:
                line = f.readline()
                l_line.append(line)

                if line == "":
                    i += 1

                if i > 100:
                    break

        except UnicodeEncodeError:
            f = open(FILE, 'r', encoding="utf_8")

            while True:
                line = f.readline()
                l_line.append(line)

                if line == "":
                    i += 1

                if i > 100:
                    break

    Parameter = {}
    Read = False
    P = None
    l_TEXT = []
    i = 0

    for line in l_line:
        line = str(line)
        line = line.replace("\t", "   ")

        if Read == False:
            for T in PARAMETER_TYPE:
                if T in line:
                    Read = True
                    l_TEXT.append(line)
                    P = ParameterName(line)
                    i = 0
        else:
            l_TEXT.append(line)

            A = line.replace(" ", "")
            A = A.replace("\n", "")

            if A == "END":
                if l_TEXT != []:
                    Parameter[P] = l_TEXT
                Read = False
                P = None

                l_TEXT = []
    f.close()

    return Parameter


# ## DCM読込み（パラメータ値モード）

# In[4]:


from scipy import interpolate

import pandas as pd
import numpy
import numpy as np

import os


# In[5]:


def CheckFloat(X):
    if X != None:
        try:
            Y = float(X)
            J = True
        except ValueError:
            J = False
    else:
        J = False

    return J


def ValueList(Mode, L_Data1, L_Data2):
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Z1 = []
    Z2 = []

    if Mode == 1:
        if L_Data1 == None:
            Z1 = None
        else:
            Z1 = L_Data1[2][0]

        if L_Data2 == None:
            Z2 = None
        else:
            Z2 = L_Data2[2][0]

    elif Mode == 2:
        if L_Data1 != None:
            X1 = L_Data1[0]

        if L_Data2 != None:
            X2 = L_Data2[0]

        if L_Data1 != None:
            Z1 = L_Data1[2]

        if L_Data2 != None:
            Z2 = L_Data2[2]

    elif Mode == 3:
        if L_Data1 != None:
            X1 = L_Data1[0]

        if L_Data2 != None:
            X2 = L_Data2[0]

        if L_Data1 != None:
            Y1 = L_Data1[1]

        if L_Data2 != None:
            Y2 = L_Data2[1]

        if L_Data1 != None:
            Z1 = L_Data1[2]

        if L_Data2 != None:
            Z2 = L_Data2[2]

    return X1, X2, Y1, Y2, Z1, Z2


def ParameterValue(Target, Text):
    OUT_temp = []

    L_Text = Text.split(" ")
    i = 0
    for T in L_Text:
        if T == Target:
            OUT_temp = L_Text[i + 1 :]
            break
        i += 1

    OUT = []
    for O in OUT_temp:
        if O != "":
            try:
                OUT.append(float(O))
            except ValueError:
                OUT.append(O)

    return OUT


def PramameterToXYZ(L_Text):
    X = []
    Y = []
    Z = []
    for T in L_Text:
        T = T.replace("\n", "")
        T = T.replace("\t", "")

        if "ST/X" in T:
            X += ParameterValue("ST/X", T)
        elif "ST/Y" in T:
            Y += ParameterValue("ST/Y", T)
        elif "WERT" in T:
            Z += ParameterValue("WERT", T)
        elif "TEXT" in T:
            Z += ParameterValue("TEXT", T)

    return [X, Y, Z]


def ParameterValueFromDCM(File):
    D_Parameter = {}
    TEXT = []
    READ = False

    TYPE = ["FESTWERT", "KENNLINIE", "KENNFELD"]

    try:
        f = open(File, "r")
        datalist = f.readlines()
    except UnicodeDecodeError:
        try:
            f = open(File, "r", encoding="shift-jis")
            datalist = f.readlines()
        except UnicodeEncodeError:
            f = open(File, "r", encoding="utf_8")
            datalist = f.readlines()

    for data in datalist:
        for T in TYPE:
            if T in data:
                READ = True

                X = data.split(" ")
                Parameter = X[1].replace("\n", "")

        if READ == True:
            TEXT.append(data)

            X = data.replace(" ", "")
            X = X.replace("\n", "")
            if X == "END":
                READ = False
                D_Parameter[Parameter] = TEXT
                TEXT = []

    f.close()

    return D_Parameter


def Read_DCM_wo_Text(File):
    d_Parameter = {}

    d_Parameter_1 = ParameterValueFromDCM(File)

    for Target in d_Parameter_1:
        d_Parameter[Target] = PramameterToXYZ(d_Parameter_1[Target])

    return d_Parameter


# ## Read_DCM

# In[4]:


def Read_DCM(File):
    d_Out = {}
    
    d_P = Read_DCM_w_Text(File)
    d_P_wo = Read_DCM_wo_Text(File)
    
    for P in d_P.keys():
        _ = d_P_wo[P]
        X = _[0]
        Y = _[1]
        Z = _[2]
        T = d_P[P][0]

        # if X != [] and Y != []:
        #     Type = "KENNFELD"
        # elif X != []:
        #     Type = "KENNLINIE"
        # else:
        #     Type = "FESTWERT"

        for F in ["FESTWERT TEXT", "FESTWERT", "KENNLINIE", "KENNFELD"]:
            if F in T:
                Type = F
            
        d_Out[P] = (Type, X, Y, Z, d_P[P])
        
    return d_Out


# ## Read_Value_1D

# In[7]:


def Read_Value_1D(_):
    X, Y, Z, T = _
    return Z[0]


# ## Read_Value_2D

# In[8]:


def Read_Value_2D(_, x_):
    X, Y, Z, T = _
    
    if len(Z) > 1:
        if x_ > X[-1]:
            value = X[-1]
        elif x_ < X[0]:
            value = X[0]
        else:
            value = x_

        f = interpolate.interp1d(X, Z)
        Out = f(value)
    else:
        Out = Z[0]
    
    return Out


# ## Read_Value_3D

# In[96]:


def Read_Value_3D(_, value):
    X, Y, Z, T = _
    x_, y_ = value
    # print(Z)
    
    # X_, Y_ = np.meshgrid(X, Y)
    # Z_ = np.reshape(Z, (len(X), len(Y)))
    # print(Z_)
    X_ = X
    Y_ = Y
    Z_ = np.reshape(Z, (len(Y), len(X)))
    
    if x_ > X[-1]:
        value_x = X[-1]
    elif x_ < X[0]:
        value_x = X[0]
    else:
        value_x = x_

    if y_ > Y[-1]:
        value_y = Y[-1]
    elif y_ < Y[0]:
        value_y = Y[0]
    else:
        value_y = y_
    
    f = interpolate.interp2d(X_, Y_, Z_)
    Out = f(value_x, value_y)
    
    return Out[0]


# In[97]:


# X = ([0.1201171875, 0.2001953125, 0.349609375, 0.5, 0.7998046875], [0.080078125, 0.1201171875, 0.2001953125, 0.349609375, 0.7001953125], [0.28125, 0.359375, 0.40625, 0.5, 0.5, 0.1875, 0.234375, 0.3125, 0.46875, 0.5, 0.140625, 0.1875, 0.265625, 0.421875, 0.5, 0.140625, 0.171875, 0.234375, 0.40625, 0.5, 0.140625, 0.15625, 0.203125, 0.296875, 0.296875], ['KENNFELD TCSOp_TCSStart_vTarOffset_wMu 5 5\n', '   LANGNAME "<unknown>"\n', '   EINHEIT_X ""\n', '   EINHEIT_Y ""\n', '   EINHEIT_W ""\n', '   ST/X  0.1201171875  0.2001953125  0.349609375  0.5  0.7998046875\n', '   ST/Y  0.080078125\n', '   WERT  0.28125  0.359375  0.40625  0.5  0.5\n', '   ST/Y  0.1201171875\n', '   WERT  0.1875  0.234375  0.3125  0.46875  0.5\n', '   ST/Y  0.2001953125\n', '   WERT  0.140625  0.1875  0.265625  0.421875  0.5\n', '   ST/Y  0.349609375\n', '   WERT  0.140625  0.171875  0.234375  0.40625  0.5\n', '   ST/Y  0.7001953125\n', '   WERT  0.140625  0.15625  0.203125  0.296875  0.296875\n', 'END\n'])
# # Y = (0.416992, 0.00195313)
# Y = (0.8, 0.7)

# Z = Read_Value_3D(X, Y)


# In[101]:


# DCM = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\rev220_Complete_ESP10CB_VarCode_1.dcm'
# DCM_new = 'c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\rev230_Complete_ESP10CB_VarCode_1.dcm'
# DIR = ['c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\All']
# # DIR = ['c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\High']
# # DIR = ['c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\Low']
# # DIR = ['c:\\TSDE_Workarea\\ktt2yk\\Work\\Act_Lamp\\3T0A_20231023_FromIgarashisan\\Select']

# d_PARAMETER_new = Read_DCM(DCM_new)

# vx = 12
# ay = 5
# P_MinSevearityPtVDCOS = Read_Value_3D(d_PARAMETER_new['VDCOp_AHA_MinSevearityPtVDCOS'], (vx, abs(ay)))
# print(P_MinSevearityPtVDCOS)


# # Compare_DCM

# ## DiffCheck
# - 差分評価(分解能違いを判定)
# - 1.5%未満は差分無しと判断する。

# In[10]:


def DiffCheck(Mode, L_Data1, L_Data2):
    X1, X2, Y1, Y2, Z1, Z2 = ValueList(Mode, L_Data1, L_Data2)

    if Mode == 1:
        J = DiffCheckValue(Z1, Z2)

    elif Mode == 2:
        if DiffCheckLenValue(X1, X2) == False and DiffCheckLenValue(Z1, Z2) == False:
            J = False
        else:
            J = True

    elif Mode == 3:
        if DiffCheckLenValue(X1, X2) == False and DiffCheckLenValue(Y1, Y2) == False and DiffCheckLenValue(Z1, Z2) == False:
            J = False
        else:
            J = True

    return J


def CheckFloat(X):
    if X != None:
        try:
            Y = float(X)
            J = True
        except ValueError:
            J = False
    else:
        J = False

    return J


def DiffCheckValue(X1, X2):
    J = True

    if X1 == X2:
        J = False
    else:
        if X1 == 0 or X2 == 0:
            J = True
        else:
            if CheckFloat(X1) and CheckFloat(X2):
                if abs((X2 - X1) / X1) < 0.015 and abs((X2 - X1) / X2) < 0.015:
                    J = False

        if CheckFloat(X1) and CheckFloat(X2):
            if float(X1) == float(X2):
                J = False

    return J


def DiffCheckLenValue(X1, X2):
    J = True

    if len(X1) == len(X2):
        n = len(X1)
        for i in range(n):
            J_temp = DiffCheckValue(X1[i], X2[i])
            if J_temp == True:
                break

        if J_temp == False:
            J = False

    return J


def ValueList(Mode, L_Data1, L_Data2):
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Z1 = []
    Z2 = []

    if Mode == 1:
        if L_Data1 == None:
            Z1 = None
        else:
            Z1 = L_Data1[2][0]

        if L_Data2 == None:
            Z2 = None
        else:
            Z2 = L_Data2[2][0]

    elif Mode == 2:
        if L_Data1 != None:
            X1 = L_Data1[0]

        if L_Data2 != None:
            X2 = L_Data2[0]

        if L_Data1 != None:
            Z1 = L_Data1[2]

        if L_Data2 != None:
            Z2 = L_Data2[2]

    elif Mode == 3:
        if L_Data1 != None:
            X1 = L_Data1[0]

        if L_Data2 != None:
            X2 = L_Data2[0]

        if L_Data1 != None:
            Y1 = L_Data1[1]

        if L_Data2 != None:
            Y2 = L_Data2[1]

        if L_Data1 != None:
            Z1 = L_Data1[2]

        if L_Data2 != None:
            Z2 = L_Data2[2]

    return X1, X2, Y1, Y2, Z1, Z2


# ## Compare_DCM

# In[11]:


def Check_Mode(l_):
    if len(l_[0]) != 0 and len(l_[1]) != 0:
        Mode = 3
    elif len(l_[0]) != 0:
        Mode = 2
    else:
        Mode = 1
    
    return Mode


# In[12]:


def Compare_DCM(DCM1, DCM2):
    d_Diff = {}
    
    d_DCM1 = Read_DCM(DCM1)
    d_DCM2 = Read_DCM(DCM2)
    
    Parameter = list(d_DCM1.keys()) + list(d_DCM2.keys())
    Parameter = list(set(Parameter))
    Parameter.sort()
    
    for P in Parameter:
        if '.' not in P:
            Value1 = None
            Value2 = None

            if P in d_DCM1:
                _ = d_DCM1[P]
                X, Y, Z, T = _
                Value1 = [X, Y, Z]

            if P in d_DCM2:
                _ = d_DCM2[P]
                X, Y, Z, T = _  
                Value2 = [X, Y, Z]
            
            if Value1 != Value2 and Value1 != None and Value2 != None:
                Mode1 = Check_Mode(Value1)
                Mode2 = Check_Mode(Value2)
                
                if Mode1 == Mode2:
                    if DiffCheck(Mode1, Value1, Value2) == True:
                        d_Diff[P] = (Value1, Value2)
                else:
                    d_Diff[P] = (Value1, Value2)
            
            if Value1 == None or Value2 == None:
                d_Diff[P] = (Value1, Value2)
    
    return d_Diff


# In[13]:


# DCM1 = '/Work/Traces/3T0A/20230221_Winter/DCM/Base/rev212_Complete_ESP10CB_VarCode_1.dcm'
# DCM2 = '/Work/Traces/3T0A/20230221_Winter/DCM/New/20230221_0014_SharCC_PMSe.par.txt'
# d_Out = Compare_DCM(DCM1, DCM2)


# # PLOT_3D

# In[14]:


def RangeXYZ(Range):
    x_min, x_max, y_min, y_max, z_min, z_max = Range
    
    Out = (x_min_, x_max_, y_min_, y_max_, z_min_, z_max_)
    
    return Out


# In[15]:


def PLOT_3D(F, Parameter):
    d_ = Read_DCM(F)
    
    x, y, z, _ = d_[Parameter]
    # print(d_[Parameter])
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(z, (len(y), len(x)))
    
    # if Range == None:
    #     x_min = min(x)
    #     x_max = max(x)
    #     y_min = min(y)
    #     y_max = max(y)
    #     z_min = min(z)
    #     z_max = max(z)
    #     N = 10
    
    fig = plt.figure(figsize=(4, 2), dpi=100)

    # cb_min, cb_max = range_
    # cb_div = 20
    if min(z) != max(z):
        interval_of_cf = np.linspace(min(z), max(z), 10)
    else:
        interval_of_cf = [z[0]-1, z[0]+1]
    # contour_filled = plt.contourf(X, Y, Z, interval_of_cf)

    # print(interval_of_cf)
    contour_filled = plt.contourf(X, Y, Z, interval_of_cf)
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    # contour_filled = plt.contourf(X, Y, Z)
    plt.colorbar(contour_filled)
    plt.title(Parameter)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# In[16]:


# FILE = '/Work/Traces/3T0A/20230221_Winter/DCM/New/20230221_0006_SharCC_PMSe.par.txt'
# FILE2 = '/Work/Traces/3T0A/20230221_Winter/DCM/Base/rev212_Complete_ESP10CB_VarCode_1.dcm'
# PARAMETER = 'VDC_FF_APP_CoMueRatio_NoKey'

# PLOT_3D(FILE2, PARAMETER)


# # Remove_File

# In[6]:


def Remove_File(FILE, Loop=False):
        if Loop == False:
            if os.path.exists(FILE):
                try:
                    os.remove(FILE)
                except PermissionError:
                    print(f'PermissionError: {FILE} can not be removed.')
        else:
            while os.path.exists(FILE):
               try:
                   os.remove(FILE)
               except PermissionError:
                    print(f'PermissionError: {FILE} can not be removed.')
                    time.sleep(10)


# In[9]:


def Remove_Folder(folder_path, Loop=False):
    if Loop == False:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
            except PermissionError:
                    print(f'PermissionError: {folder_path} can not be removed.')
    else:
        while os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
            except PermissionError:
                print(f'PermissionError: {folder_path} can not be removed.')
                time.sleep(10)


# In[ ]:




