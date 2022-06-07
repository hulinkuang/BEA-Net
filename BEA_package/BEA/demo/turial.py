import os
from BEA.demo.preprocess import preprocess, output_folder


if __name__ == "__main__":
    train = True
    # 1. 修改nnUNet path文件中的路径
    # 2. 修改preprocess 文件中的路径

    preprocess()  # only run one time

    # Training
    # Also cd BEA_package/BEA/run
    # Run nohup python run_training.py -gpu='0' -outpath='BEA' -task 026 2>&1 & for training.
    if train:
        os.system('/home/wyh/anaconda3/envs/BSG/bin/python -u /homeb/wyh/Codes/BEA-Net/BEA_package/BEA/run/run_training.py'
                  ' -gpu=\'0\' -outpath=\'BEA\' -task 026')

    # Testing
    # Also cd BEA_package/BEA/run
    # Run nohup python run_training.py -gpu='0' -outpath='BEA' -task 026 -val 2>&1 & for training.
    else:
        os.system(
            '/home/wyh/anaconda3/envs/BSG/bin/python -u /homeb/wyh/Codes/BEA-Net/BEA_package/BEA/run/run_training.py'
            ' -gpu=\'0\' -outpath=\'BEA\' -task 026 -val')
