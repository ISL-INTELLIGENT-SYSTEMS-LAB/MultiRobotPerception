'''
This file accepts output csv files from the ZED camera, then uses the class confidence and
3d bounding box information to create a relation between those detected objects. This is done by
using a box and class loss method defined by the DETR model. Then, the hungarian algorithm is run
on the resulting matrix to give a relation with the lowest cost. Such is returned

@file network_relation.py
@version 11/2/2023
@author Intelligent Systems Laboratory @ UNCFSU
'''

import os
import numpy as np
import pandas as pd
import time
from utils import *

class InputFileException(Exception):
        """ This class is run when there is an issue with the input files """


class NetworkRelation:
        def __init__(self, root, files):
                self.root = root
                self.files = files


        def get_data_frames(self):
                data_frames = [get_data(self.root, file) for file in self.files]
                max_objects_detected = max(len(df['Object']) for df in data_frames)
                for df in data_frames:
                        while df.shape[0] != max_objects_detected:
                                df.loc[len(df.index)] = [None] * len(df.loc[0])
                return data_frames, max_objects_detected

        def network_relation(self):
                start = time.time()
                count = 0
                data_frames, max_objects_detected = self.get_data_frames()
                for file in self.files:
                        if not file.endswith('.csv'):
                                raise InputFileException("Input files must all be of type .csv")
                while len(data_frames) > 1:
                        df_1 = data_frames[0]
                        df_2 = data_frames[1]
                        temp = pd.DataFrame(float('inf'), index=df_2['Object'].to_list(), columns=df_1['Object'].to_list())
                        df_1_object_list = [item for i in range(len(df_1)) if df_1['Class Confidence'][i] is not None and df_1['3D_Bounding_Box'][i] is not None
                                                                for item in [[df_1['Class Confidence'][i], df_1['3D_Bounding_Box'][i]]]]
                        df_2_object_list = [item for j in range(len(df_2)) if df_2['Class Confidence'][j] is not None and df_2['3D_Bounding_Box'][j] is not None
                                                                for item in [[df_2['Class Confidence'][j], df_2['3D_Bounding_Box'][j]]]]
                        for key1, node1 in enumerate(df_1_object_list):
                                for key2, node2 in enumerate(df_2_object_list):
                                        temp.at[df_2['Object'][key2], df_1['Object'][key1]] = get_loss_2box(node1[1], node1[0], node2[1], node2[0])
                        temp.replace(float('inf'), np.finfo(np.float64).max, inplace=True)
                        cols, rows, _ = hungarian(temp.to_numpy())
                        valid_assignments = temp.to_numpy()[cols, rows] != np.finfo(np.float64).max
                        cols = cols[valid_assignments]
                        rows = rows[valid_assignments]
                        correspondence_df = pd.DataFrame({self.files[count].split('.csv')[0]: cols, self.files[count+1].split('.csv')[0]: rows})
                        correspondence_df.index = ['Person_' + str(i) for i in range(len(correspondence_df))]
                        data_frames = data_frames[1:]
                        count += 1
                        print(correspondence_df)
                end = time.time()
                print(f"Total time: {end-start}")
                print(correspondence_df)
                correspondence_df.to_csv(self.root + 'output_relation.csv')
                return [temp.to_numpy()[cols, rows].sum(), cols]

        def main(self):
                experiment_name = r'Exp1A/'
                final_filepath = os.path.join(self.root, experiment_name)
                self.files = [filename for filename in os.listdir(final_filepath) if (filename.endswith('.csv') and not filename.startswith('output'))]
                print(self.network_relation())

if __name__ == '__main__':
    root = "/path/to/your/directory/"  # replace with your actual directory
    files = []  # initially empty, will be filled in main method
    nr = NetworkRelation(root, files)
    nr.main()