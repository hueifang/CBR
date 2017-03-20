echo "Downloading ImageNet pre-trained model and weight"
mkdir -p ./models/bvlc_reference_caffenet
wget -O ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel "https://www.dropbox.com/s/nlggnj47xxdmwkb/bvlc_reference_caffenet.caffemodel?dl=1"

echo "Downloading CIFAR-10 Dataset"
wget -O cifar10-dataset.zip "https://www.dropbox.com/s/f7q3bbgvat2q1u2/cifar10-dataset.zip?dl=1" 
unzip cifar10-dataset.zip -d ./data/cifar-10
rm cifar10-dataset.zip