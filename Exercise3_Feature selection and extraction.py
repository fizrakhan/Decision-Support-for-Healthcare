# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:31:47 2023

@author: HP
"""

import pandas as pd
import numpy as np

experiment = 1
subject =1
activity_label = 5

file_path = f"C:/Users/HP/Documents/Master's/Decision Support/Exercise 3/RawData/acc_exp0{experiment}_user0{subject}.txt"
acc_user00 = pd.read_csv(file_path, index_col= False, names = ['x', 'y', 'z'], sep = " " )
labels = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise 3/RawData/labels.txt", index_col=False, names = ["Test", "Subject", "Label", "Start", "Stop"], sep= " ")

user00_data = np.where([labels['Subject'] == subject])
user00_data = np.where([labels['Test'] == subject])
user00_data = np.where(labels.loc[labels['Label'] == activity_label])
user00_data = np.where(labels.loc['Start'] == subject)
user00_data = np.where(labels.loc['Stop'] == subject)


