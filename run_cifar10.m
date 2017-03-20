% demo code that evaluates the retrieval performance of a trained model on CIFAR10
clear;

%% -- settings start here ---

% path to the caffe
addpath('/path/to/caffe/matlab');

% helper functions
addpath('./utils');

feat_len = 512;
IMAGE_DIM = 256; % resize to image_dim x image_dim
CROPPED_DIM = 227; % crop to cropped_dim x cropped_dim. 224 for VGG16; 227 for AlexNet

mode = 1; % 1 for dot product; 2 for Euclidean distance

% set result foldert
result_folder =  './results/cifar-10/round_2';
% the place where lists of training files and of test files are placed
data_folder = './data/cifar-10/';
% the model
net_weights = './experiments/cifar-10/round_2/models/CBR_iter_50000.caffemodel';
% model definition
net_model = './experiments/cifar-10/round_2/cfgs/deploy_abstract.prototxt';

%% --- feature extraction ---

% get lists of training and test samples as well as their labels
test_file_list = sprintf('%s/test-file-list.txt', data_folder);
test_label_file = sprintf('%s/test-label.txt', data_folder);
train_file_list = sprintf('%s/train-file-list.txt', data_folder);
train_label_file = sprintf('%s/train-label.txt', data_folder);
                                
if ~exist(result_folder, 'file')
    mkdir(result_folder);
end
                
feat_test_file = sprintf('%s/test.mat', result_folder);
feat_train_file = sprintf('%s/train.mat', result_folder);
                
train.im_list = read_cell (train_file_list);
train.label = load(train_label_file);
                
% extract features of training images
train.feat = feat_batch (1, net_model, net_weights, train.im_list, feat_len, IMAGE_DIM, CROPPED_DIM);
save(feat_train_file, 'train', '-v7.3');
                
test.im_list = read_cell (test_file_list);
test.label = load(test_label_file);
                
% extract features of test images
test.feat = feat_batch (1, net_model, net_weights, test.im_list, feat_len, IMAGE_DIM, CROPPED_DIM);   
save(feat_test_file, 'test','-v7.3');

%% --- compute mAP and precition at k ---
map_file = [result_folder '/map.txt'];
precision_file = [result_folder '/precision-at-k.txt'];

train.feat = train.feat';
test.feat = test.feat';

[map, precision_at_k] = map_precision( train.label, train.feat, ...
    test.label, test.feat, mode);
                    
% save results
outfile = fopen(map_file, 'w');
fprintf(outfile, '%.4f\t', map);
fclose(outfile);

P = [[1:1:size(precision_at_k,2)]' precision_at_k'];
save(precision_file, 'P', '-ascii');      