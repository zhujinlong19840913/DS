# Dynamic Sampling Strategy for Enhanced Person Re-Identification across Multiple Cameras

Code for the paper [Dynamic Sampling Strategy for Enhanced Person Re-Identification across Multiple Cameras](https://authors.elsevier.com/a/1l4oF3PiGTXHXU).

**This code is ONLY** released for academic use.

## Installation

This codebase has been developed with python version 3.10, PyTorch version 2.1.2, CUDA 12.1 and torchvision 0.16.2.                                           

## Datasets

We use Market-1501 and MSMT17 as our training data.

## Prepare Pre-trained Models 
You can download models from dynamicSampler, or use dynamicSampler to train your own models.
Before training, you should convert the models first.   

```bash
python convert_model.py path/to/dynamicSampler/swin_base.pth path/to/dynamicSampler/checkpoint_tea.pth
```

## Training

We utilize 1 GPU for training. Please modify the `MODEL.PRETRAIN_PATH`, `DATASETS.ROOT_DIR` and `OUTPUT_DIR` in the config file.

```bash
sh run.sh
```

## Test

```bash
sh runtest.sh
```

## Performance

| Method | MSMT17 | Market1501 |
| ------ | :---: | :---: |
| dynamicSampler | 76.5/90.7 | 94.1/97.1  |

- `mAP/Rank1` are used as evaluation metric.
- The semantic weight is set to 0.2 in these experiments.

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{zhu2025dynamic,
  title={Dynamic Sampling Strategy for Enhanced Person Re-Identification across Multiple Cameras},
  author={Jinlong Zhu and Wanping Yang and Qingliang Li and Yuguang Yan},
  booktitle={Expert Systems with Applications},
  year={2025},
}
```
