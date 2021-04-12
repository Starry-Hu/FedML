#!/usr/bin/env sh
 rm -rf mnist
 rm -rf train
 rm -rf test
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
# --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf' -O- \
# | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf" -O MNIST.zip && rm -rf /tmp/cookies.txt


 unzip MNIST.zip

 mv mnist/train train
 mv mnist/test test
 rm -rf mnist
 rm -rf MNIST.zip


# This scripts downloads the mnist data and unzips it.【解决上述google访问失败的备用方法】
# caffe下载MNIST数据的脚本
#DIR="$( cd "$(dirname "$0")" ; pwd -P )"
#
#cd "$DIR"
#echo $DIR
#echo "Downloading..."
#
#rm -rf mnist
#rm -rf train
#rm -rf test
#
#
#for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
#do
#  if [ ! -e $fname ]; then
#    wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
#    gunzip ${fname}.gz
#  fi
#done
#
#mv mnist/train train
#mv mnist/test test
#rm -rf mnist
#rm -rf MNIST.zip
