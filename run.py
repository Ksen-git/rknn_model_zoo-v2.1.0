cd /home/orangepi/Desktop/rk3588/rknn_model_zoo-v2.1.0
git checkout master
git checkout pipeline
git checkout pipeline-rga

git commit -m "master/main.cc"

cd /home/orangepi/Desktop/rk3588/rknn_model_zoo-v2.1.0 #需要先进入仓库根目录

# 切分支+编译+运行 一条命令搞定
git checkout master && \
cd examples/yolov8/cpp/build && \
cmake .. -DTARGET_SOC=rk3588 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release && \
make -j4 && \
./rknn_yolov8_demo /home/orangepi/Desktop/rk3588/rknn_models/yolov8_int8_rk3588.rknn /home/orangepi/Desktop/rk3588/inputimg/COCO-test1.jpg

cd /home/orangepi/Desktop/rk3588/rknn_model_zoo-v2.1.0
# 切分支+编译+运行 一条命令搞定
git checkout pipeline && \
cd examples/yolov8/cpp/build && \
cmake .. -DTARGET_SOC=rk3588 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release && \
make -j4 && \
./rknn_yolov8_demo /home/orangepi/Desktop/rk3588/rknn_models/yolov8_int8_rk3588.rknn /home/orangepi/Desktop/rk3588/inputimg/COCO-test1.jpg 100

cd /home/orangepi/Desktop/rk3588/rknn_model_zoo-v2.1.0
# 切分支+编译+运行 一条命令搞定
cd /home/orangepi/Desktop/rk3588/rknn_model_zoo-v2.1.0 && \
git checkout pipeline-rga && \
cd examples/yolov8/cpp/build && \
cmake .. -DTARGET_SOC=rk3588 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release && \
make -j4 && \
./rknn_yolov8_demo /home/orangepi/Desktop/rk3588/rknn_models/yolov8_int8_rk3588.rknn /home/orangepi/Desktop/rk3588/inputimg/COCO-test1.jpg 100










