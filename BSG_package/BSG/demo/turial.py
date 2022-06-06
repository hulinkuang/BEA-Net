import os
from BSG.demo.preprocess import preprocess, output_folder


if __name__ == "__main__":
    train = False
    # 1. 修改nnUNet path文件中的路径
    # 2. 修改preprocess 文件中的路径
    if not output_folder.exists():
        preprocess()

    # Training
    # Also cd BSG_package/BSG/run
    # Run nohup python run_training.py -gpu='0' -outpath='BSG' -task 026 2>&1 & for training.
    if train:
        os.system('/home/kuanghl/anaconda3/envs/BSG/bin/python -u /home/kuanghl/Codes/BSG/BSG_package/BSG/run/run_training.py'
                  ' -gpu=\'0\' -outpath=\'BSG\' -task 026')

    # Testing
    # Also cd BSG_package/BSG/run
    # Run nohup python run_training.py -gpu='0' -outpath='BSG' -task 026 -val 2>&1 & for training.
    else:
        os.system(
            '/home/kuanghl/anaconda3/envs/BSG/bin/python -u /home/kuanghl/Codes/BSG/BSG_package/BSG/run/run_training.py'
            ' -gpu=\'0\' -outpath=\'BSG\' -task 026 -val')
