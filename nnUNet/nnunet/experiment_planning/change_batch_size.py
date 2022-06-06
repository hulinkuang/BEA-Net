from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/kuanghl/Codes/BSG/nnUNet/nnU_data/nnUNet_raw_data_base/processed_data/Task022_IDRID/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_LISA_plans_3D.pkl'
    a = load_pickle(input_file)
    a['plans_per_stage'][0]['batch_size'] = 16
    save_pickle(a, input_file)