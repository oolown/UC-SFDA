# UC-SFDA

The proposed UC-SFDA is programmed in PyTorch 1.9.0 and trained on a server with Intel(R) Core(TM) i9-12900K CPU and NVIDIA GeForce RTX3090 GPU.

The training log file and the model file are available by linking ：https://pan.baidu.com/s/1EhKviVvBHOLKPhysCGYdfg?pwd=SFDA code：SFDA .

## Prepare pretrain model：

Download the pre-trained model from URL https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz  to path “./model/vit_checkpoint/imagenet21k”

## Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), from the official websites, and modify the path of images in each '.txt' under the folder './data/'.

## Source Training:

Train model on the source domain(The default task is domain **A** in OFFICE-31, and the default parameter configuration is the recommended configuration):

```
python source_pretrain.py
```

## Target Adaptation:

Adaptation to other target domains (The default source domain is **A** and the target domain is **D**, and the default parameter configuration is the recommended configuration):

```
python target_adaptation.py
```

