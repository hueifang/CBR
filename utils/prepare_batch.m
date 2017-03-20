% ------------------------------------------------------------------------
function images = prepare_batch(image_files,mean_data,batch_size,IMAGE_DIM,CROPPED_DIM)
% ------------------------------------------------------------------------
if nargin < 2
    % mean file
    mean_data(:,:,1) = ones(IMAGE_DIM,IMAGE_DIM)*104;
    mean_data(:,:,2) = ones(IMAGE_DIM,IMAGE_DIM)*117;
    mean_data(:,:,3) = ones(IMAGE_DIM,IMAGE_DIM)*123;
end

num_images = length(image_files);
if nargin < 3
    batch_size = num_images;
end

indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;
num_images = length(image_files);
images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

parfor i=1:num_images
    % read file
    fprintf('%c Preparing %s\n',13,image_files{i});
    try
    im = imread(image_files{i});
    % resize to fixed input size
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    % Transform GRAY to RGB
    if size(im,3) == 1
        im = cat(3,im,im,im);
    end
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]) - mean_data;
    % Crop the center of the image
    images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
    center:center+CROPPED_DIM-1,:),[2 1 3]);
    catch
        warning('Problems with file',image_files{i});
    end
end