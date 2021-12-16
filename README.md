# Efficient Cross-Modal Retrievalvia Deep Binary Hashing and Quantization

This is our official implementation for the paper:

Yang Shi and Young-joo Chung. 2021. <a href="https://www.bmvc2021-virtualconference.com/assets/papers/1202.pdf">Efficient Cross-Modal Retrievalvia Deep Binary Hashing and Quantization</a>. In Proceedings of the 32nd British Machine Vision Conference (BMVC ’21), Nov 22–25, 2021, Virtual Event, 14 pages.


In this work, we propose a jointly learned deep hashing and quantization network (HQ) for cross-modal retrieval. We simultaneously learn binary hash codes and quantization codes to preserve semantic information in multiple modalities by an end-to-end deep learning architecture. At the retrieval step, binary hashing is used to retrieve a subset of items from the search space, then quantization is used to re-rank the retrieved items. We theoretically and empirically show that this two-stage retrieval approach provides faster retrieval results while preserving accuracy. Experimental results on three datasets demonstrate that HQ achieves boosts of more than 7% in precision compared to supervised neural network-based compact coding models.

<p align="center">
   <b>Training</b><br>
<img src="training.pdf" width="500">
 </p>
 
<p align="center">
  <b>Retrieval</b><br>
  <img src="retrieval.pdf" width="500">
</p>


## Environment Settings
We use Pytorch as the backend.
- Python 3.8.3
- Pytorch version:  '1.7.1'

## Example to run the codes.
We use pretrained AlexNet model for the image backbone. Before running the codes, you can run 

```
cp ./alexnet-owt-4df8aa71.pth /root/.cache/torch/hub/checkpoints/
```

to add the pretrained model in the root.

Once you have it, you can run

```
python main.py --dataset nuswide --feature_dim 100
```


### Dataset
We provide the files to run our method on the <a href="https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html">NUS-WIDE</a> and  <a href="https://press.liacs.nl/researchdownloads/">MIR-Flickr</a> datasets. Due to the size limit, the provided datasets are sampled from the original datasets.


