from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/homeb/wyh/Codes/CoTr_KSR/nnUNet/nnU_data/nnUNet_raw_data_base/processed_data/Task018_ISIC/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_LISA_plans_3D.pkl'
    a = load_pickle(input_file)
    a['plans_per_stage'][0]['batch_size'] = 32
    save_pickle(a, input_file)
