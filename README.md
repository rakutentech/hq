# Efficient Cross-Modal Retrievalvia Deep Binary Hashing and Quantization

This is our official implementation for the paper:

Yang Shi and Young-joo Chung. 2021. <a href="https://www.bmvc2021-virtualconference.com/assets/papers/1202.pdf">Efficient Cross-Modal Retrievalvia Deep Binary Hashing and Quantization</a>. In Proceedings of the 32nd British Machine Vision Conference (BMVC ’21), Nov 22–25, 2021, Virtual Event, 14 pages.


In this work, we propose a jointly learned deep hashing and quantization network (HQ) for cross-modal retrieval. We simultaneously learn binary hash codes and quantization codes to preserve semantic information in multiple modalities by an end-to-end deep learning architecture. At the retrieval step, binary hashing is used to retrieve a subset of items from the search space, then quantization is used to re-rank the retrieved items. We theoretically and empirically show that this two-stage retrieval approach provides faster retrieval results while preserving accuracy. Experimental results on three datasets demonstrate that HQ achieves boosts of more than 7% in precision compared to supervised neural network-based compact coding models.

The training and retrieval process are summarized here.


Training            |  Retrieval
:-------------------------:|:-------------------------:
![](./training.png)  |  ![](./retrieval.png)


## Environment Settings
We use Pytorch as the backend.
- Python 3.8.3
- Pytorch version:  '1.7.1'

## Example to run the codes.
We use pretrained AlexNet model for the image backbone. Before running the codes, you can run 

```
cp ./alexnet-owt-4df8aa71.pth /root/.cache/torch/hub/checkpoints/
```

to add the pretrained model in the root. alexnet-owt-4df8aa71.pth can be downloaded from <a href="https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth">here</a>.

Once you have it, you can run

```
python main.py --dataset nuswide --feature_dim 100
```


### Dataset
Due to the size limit, the datasets <a href="https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html">NUS-WIDE</a> and  <a href="https://press.liacs.nl/researchdownloads/">MIR-Flickr</a> are not provided in this repository. The data format used in this repository should be: 

- xxxx_image.npy stores image raw or pretrained features. it is of size N-by-h-by-w-by-c, where N is the number of samples, h,w, and c are the height, width and chanel size of the image. 

- xxxx_text.npy stores text one-hot embeddings. It is of size N-by-l, where N is the number of samples, l is the vocabulary size.  

- xxxx_label.npy stores label one-hot embeddings. It is of size N-by-k, where N is the number of samples, k is the number of labels.   

- xxxx can be train, test or vali.

Data should be put in a folder with name "data" with subfolder "dataset". Inside ```./data/dataset/```, you should have train_image.npy, train_text.npy, train_image_label.npy, train_text_label.npy, test_image.npy, test_text.npy, test_label.npy.

