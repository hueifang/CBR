#!/bin/sh

# path to the place where caffe is installed
CAFFE_BIN=/path/to/caffe/build/tools/caffe 

# train an initial model with a classification loss
ILSVRC_Weight=./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

# create a folder for round 0
mkdir -p ./experiments/cifar-10/round_0/models
mkdir -p ./experiments/cifar-10/round_0/cfgs

# copy train_val_round0.prototxt and solver_round0.prototxt from template to the folder of round 0
cp ./template/train_val_round0.prototxt ./experiments/cifar-10/round_0/cfgs/train_val.prototxt
cp ./template/solver_round0.prototxt ./experiments/cifar-10/round_0/cfgs/solver.prototxt

# train the initial model
LOGFILE=./experiments/cifar-10/round_0/models/log.txt
$CAFFE_BIN train -solver ./experiments/cifar-10/round_0/cfgs/solver.prototxt -weights $ILSVRC_Weight -gpu 0 2>&1 | tee $LOGFILE

# network configuration to be read in the python code for target code update
DEPLOY=./template/deploy.prototxt

# a list of training samples to be read in for target code update 
INPUT_IMAGES=./data/cifar-10/train-file-list.txt
# a list of class labels of training samples to be read in for target code update
INPUT_LABELS=./data/cifar-10/train-label.txt

# CBR training
NUM_CODE_UPDATES=2
i=1
while [ "$i" -le "$NUM_CODE_UPDATES" ]; do
  j=$(( i - 1 ))

  INLOGFILE="./experiments/cifar-10/round_""$j""/models/log.txt"
  WEIGHTS_IN=$(awk '/Snapshotting to/{f=$NF} END{print f}' $INLOGFILE)
  
  # create a folder for outputting the models and config files for the current round
  MODEL_DIR="./experiments/cifar-10/round_""$i""/models"
  CONFIG_DIR="./experiments/cifar-10/round_""$i""/cfgs"
  mkdir -p ${MODEL_DIR}
  mkdir -p ${CONFIG_DIR}
  
  WEIGHTS_OUT=${MODEL_DIR}/init.caffemodel
  
  # compute target codes and update the network
  python update_model.py $WEIGHTS_IN $WEIGHTS_OUT $DEPLOY $INPUT_IMAGES $INPUT_LABELS
  
  # copy config files to the folder for the current round
  for pname in train_val solver deploy deploy_abstract; do
    cp ./template/${pname}.prototxt ${CONFIG_DIR}/${pname}.prototxt
  done
  
  # train the network with the CBR criterion and the classification loss
  LOGFILE="$MODEL_DIR""/log.txt"
  sed -i "s/IDX/$i/g" ${CONFIG_DIR}/solver.prototxt
  $CAFFE_BIN train -solver ${CONFIG_DIR}/solver.prototxt -weights $WEIGHTS_OUT -gpu 0 2>&1 | tee $LOGFILE
  
  i=$(( i + 1 ))
done
