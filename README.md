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

## Train CBR on CIFAR-10
1. Download the pretrained model and CIFAR-10 dataset.

       $./download.sh
       
2. Modify the `CAFFE_BIN` in run.sh to the path where your CAFFE is installed.
3. Modify the `caffe_root` in update_model.py to the path where you CAFFE is installed.
4. Launch the script to train CBR

       $./run.sh
