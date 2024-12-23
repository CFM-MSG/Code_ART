Zero-Shot Video Moment Retrieval with Angular Reconstructive Text Embeddings
=====

This is our open-source implementation codes of "Zero-Shot Video Moment Retrieval with Angular Reconstructive Text Embeddings" (ART).  

## 1. Environment

This repository is implemented based on [PyTorch](http://pytorch.org/) with Anaconda.</br>

### Get code and environment  
Make your own environment (If you use docker envronment, you just clone the code and execute it.)
```bash
conda create --name ART --file requirements.txt
conda activate ART
```

### Working environment
RTX 3090

Ubuntu 20.04.1

## 2. Prepare data

We employ the pre-trained I3D model to extract the Charades-STA features, while C3D models extract the ActivityNet-Caption.
You can download the video, text CLIP features and pretrained model at the google drive:[Charades](https://drive.google.com/drive/folders/1bJuOrB3sWhQNyAm4GhzI9SQPxs0-wkNT](https://drive.google.com/drive/folders/1L-ALQ5yhN-aCecHbSAbYQMHHGOemQiJw?usp=sharing).  
Before training, please download the features and put them into the data/.



## 3. Training models
Using **anaconda** environment

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train \
                     --config_path config/charades/config.yml \
                     --method_type tgn_lgi \
                     --dataset charades \
                     --tag base --seed 2049
```

## 4. Evaluating pre-trained models
```bash
# Evaluate ART model trained from Charades-STA Dataset
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
                     --config config/charades/config.yml \
                     --checkpoint pretrained_model/charades/best.pkl \
                     --method tgn_lgi \
                     --dataset charades
```

The pre-trained models will report following scores.

Dataset              | R@0.3 | R@0.5 | R@0.7 | mIoU
-------------------- | ------| ------| ------| ------
Charades-STA         | 57.31 | 41.13 | 22.12 | 39.01
