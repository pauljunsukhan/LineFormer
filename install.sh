# tested for python 3.8
# conda create --name LineFormer python=3.8
#added model weights download
pip install openmim
pip install chardet
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
mim install mmcv-full
pip install scikit-image
pip install matplotlib
pip install opencv-python
pip install pillow
pip install scipy==1.9.3
pip install -e mmdetection
pip install bresenham
pip install tqdm
pip install gdown
gdown 1TJSW_IlZh3qPCxi4c7MLiegpUSPCT1jF -O iter_3000.pth
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
