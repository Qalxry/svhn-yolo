# Script for deploying YOLO training on the AutoDL cloud server
source /etc/network_turbo
cd ~/autodl-tmp
pip install opencv-python requests tqdm Pillow pandas numpy torch torchvision ultralytics albumentations matplotlib
git clone https://gh-proxy.com/https://github.com/Qalxry/svhn-yolo.git  # use gh-proxy to avoid GFW
cd svhn-yolo
chmod +x ./code/train.sh
chmod +x ./code/test.sh
./code/train.sh