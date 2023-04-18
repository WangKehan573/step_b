
import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd

def search_event(path):
    
    dir_name = []
    file_name = []
    file_size = []
    event_num = 0      
    for filename in os.listdir(path):
        
        temp_path = os.path.join(path, filename)
        if os.path.isdir(temp_path):
           
            temp_file_name = search_event(temp_path)
          
            file_name.extend(temp_file_name)
        elif os.path.isfile(temp_path):
           
            if 'tfevents' in temp_path:
                event_num += 1
                
                file_name.append(temp_path)
                file_size.append(os.path.getsize(temp_path))
    if event_num > 1:
        
        index = file_size.index(max(file_size))
        temp_file_path = file_name[index]
        if isinstance(temp_file_path, str):
            temp_file_path = [temp_file_path]
        return temp_file_path
    return file_name
    
    
def readEvent(event_path):
    
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()  
    print(event.Tags())
    scalar_name = []
    scalar_data = []
    for name in event.scalars.Keys():
        print(name)
        if 'hp_metric' not in name:
            scalar_name.append(name)
            
            scalar_data.append(event.scalars.Items(name))
    return scalar_name, scalar_data
    
    
    
def exportToexcel(file_name, excelName):
    
    writer = pd.ExcelWriter(excelName)
    for i in range(len(file_name)):
        event_path = file_name[i]
        scalar_name, scalar_data = readEvent(event_path)
        for i in np.arange(len(scalar_name)):
            scalarValue = scalar_data[i]
            scalarName = scalar_name[i]
            if "/" in scalarName:
                temp_names = scalar_name[i].split("/")
                temp_paths = os.path.split(event_path)
                scalarName = os.path.split(temp_paths[0])[1]
                print(scalarName)
            data = pd.DataFrame(scalarValue)
            data.to_excel(writer, sheet_name=scalarName)
    writer.save()
    print("save successfully")
    
    
    
def excel_to_array(excel_path, save_dir=None):
    
   
    f = pd.read_excel(excel_path, sheet_name=None)
    data_dict = dict()
    for key in f.keys():
        sheet = f[key]
        sheet = sheet.head(n=-1)
        value = sheet.values
        
        data_dict[key] = value[:, 2:4]
       
    if save_dir is not None:
        save_dict_as_mat(data_dict, save_dir)
    return data_dict

def save_dict_as_mat(dict, save_dir):
    scio.savemat(save_dir, dict)
    
    
if __name__ == "__main__":
    
    mode = 0
    eve_path = '/common-data/kehanwang/schnetpack-master/2H/forcetut/lightning_logs/version_0'
    file_name = search_event(eve_path)
    excelName = '/common-data/kehanwang/schnetpack-master/2H/1.xlsx'
    if mode == 0:
        file_name = search_event(eve_path)
        exportToexcel(file_name, excelName)
    elif mode == 1:
        excel_to_array(excelName, save_dir='your_path/loss.mat')
