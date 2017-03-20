function scores = feat_batch (use_gpu, net_model, net_weights, list_im, dim, IMAGE_DIM,CROPPED_DIM)

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

phase = 'test'; % run with phase test (so that dropout isn't applied)

if ~exist(net_weights, 'file')
    error('%s does not exist.', net_weights);
end
if ~exist(net_model, 'file')
    error('%s does not exist.', net_model);
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

% mean file
mean_data(:,:,1) = ones(IMAGE_DIM,IMAGE_DIM)*104;
mean_data(:,:,2) = ones(IMAGE_DIM,IMAGE_DIM)*117;
mean_data(:,:,3) = ones(IMAGE_DIM,IMAGE_DIM)*123;

batch_size = 10;

% prepare input
num_images = length(list_im);
scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size);
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = {prepare_batch(list_im(range),mean_data,batch_size,IMAGE_DIM,CROPPED_DIM)};
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    %output_data = caffe('forward', {input_data});  
    output_data = net.forward(input_data);
    toc
    
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);

% call caffe.reset_all() to reset caffe
caffe.reset_all();
end