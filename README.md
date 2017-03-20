# Cross-batch Reference Learning for Deep Classification and Retrieval
by Huei-Fang Yang, Kevin Lin, and Chu-Song Chen

## Introduction
This code implements the cross-batch reference (CBR) learning as described in our [ACM MM 2016 paper](http://www.iis.sinica.edu.tw/papers/song/19838-F.pdf).

## Citing the CBR
If you find CBR useful in your research, please consider citing:

    @inproceedings{yang:acmmm16,
        author    = {Huei-Fang Yang and
                     Kevin Lin and
                     Chu-Song Chen},
        title     = {Cross-batch Reference Learning for Deep Classification and Retrieval},
        booktitle = {Proc. ACM MM},
        pages     = {1237--1246},
        year      = {2016}
    }
    
## Requirements
1. `Caffe`, `matcaffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. MATLAB (required for performance evaluation)

## Train a network with CBR on CIFAR-10
1. Download the pretrained model and CIFAR-10 dataset.

       $./download.sh
       
2. Modify the `CAFFE_BIN` in run.sh to the path where the CAFFE is installed.
3. Modify the `caffe_root` in update_model.py to the path where the CAFFE is installed.
4. Launch the script to train a network with CBR. This would take a few hours.

       $./run.sh

## Evaluate the trained model
1. Modify the `addpath` in run_cifar10.m to the path where the CAFFE is installed.
2. Launch MATLAB and run the evaluation code to obtain mAP and precition@k.

       >>run_cifar10

## Contact
Please feel free to contact Huei-Fang Yang (hfyang@citi.sinica.edu.tw), Kevin Lin (kevinlin311.tw@iis.sinica.edu.tw), or Chu-Song Chen (song@iis.sinica.edu.tw) if you had any questions.
