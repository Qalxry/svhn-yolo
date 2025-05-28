source /etc/network_turbo
pip install opencv-python requests tqdm Pillow pandas numpy torch torchvision ultralytics albumentations matplotlib
git clone https://github.com/Qalxry/svhn-yolo.git
cd svhn-yolo
chmod +x ./code/train.sh
chmod +x ./code/test.sh
./code/train.sh