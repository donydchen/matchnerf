<div>
<!-- <img src='https://i.imgur.com/tFP6Q3p.gif' align="right" height="120px" width="180px" alt='house'> -->
<img src='https://i.imgur.com/Tq07diD.gif' align="right" height="120px" width="66px" alt='sculpture'> 
<img src='https://i.imgur.com/3boKX8u.gif' align="right" height="120px" width="180px" alt='printer'> 
</div>

<br><br><br><br>

# MatchNeRF




Official PyTorch implementation for MatchNeRF, a new generalizable NeRF approach that employs **explicit correspondence matching** as the geometry prior and can perform novel view synthesis on unseen scenarios with as few as two source views as input, **without requiring any retraining and fine-tuning**. <br>


>**[Explicit Correspondence Matching for Generalizable Neural Radiance Fields](http://arxiv.org/abs/2304.12294)**  
>[Yuedong Chen](https://donydchen.github.io/)<sup>1</sup>,
[Haofei Xu](https://haofeixu.github.io/)<sup>2</sup>,
[Qianyi Wu](https://qianyiwu.github.io/)<sup>1</sup>,
[Chuanxia Zheng](https://www.chuanxiaz.com/)<sup>3</sup>,
[Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/)<sup>4</sup>,
[Jianfei Cai](https://jianfei-cai.github.io/)<sup>1</sup>  
><sup>1</sup>Monash University, <sup>2</sup>ETH Zurich, <sup>3</sup>University of Oxford, <sup>4</sup>Nanyang Technological University  
arXiv 2023
### [Paper](http://arxiv.org/abs/2304.12294) | [Project Page](https://donydchen.github.io/matchnerf) | [Code](https://github.com/donydchen/matchnerf)

<img src="docs/matchnerf.png">


<details>
  <summary>Recent Updates</summary>

* `25-Apr-2023`: released MatchNeRF codes and models.

</details>

<br>


----


### Table of Contents

* [Setup Environment](#setup-environment)
* [Download Datasets](#download-datasets)
  * [DTU (for both training and testing)](#dtu-for-both-training-and-testing)
  * [Blender (for testing only)](#blender-for-testing-only)
  * [Real Forward Facing (for testing only)](#real-forward-facing-for-testing-only)
* [Testing](#testing)
* [Training](#training)
* [Rendering Video](#rendering-video)
* [Use Your Own Data](#use-your-own-data)
* [Miscellaneous](#miscellaneous)


## Setup Environment

This project is developed and tested on a **CUDA11** device. For other CUDA version, manually update the `requirements.txt` file to match the settings before preceding.

```bash
git clone --recursive https://github.com/donydchen/matchnerf.git
cd matchnerf
conda create --name matchnerf python=3.8
conda activate matchnerf
pip install -r requirements.txt
```

For rendering video output, it requires `ffmpeg` to be installed on the system, you can double check by running `ffmpeg -version`. If `ffmpeg` does not exist, consider installing it by running `conda install ffmpeg`.

## Download Datasets

### DTU (for both training and testing)

* Download the preprocessed DTU training data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth_raw.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet).

* Extract 'Cameras/' and 'Rectified/' from the above downloaded 'dtu_training.rar', and extract 'Depths' from the 'Depth_raw.zip'. Link all three folders to `data/DTU`, which should then have the following structure

```bash
data/DTU/
    |__ Cameras/
    |__ Depths/
    |__ Rectified/
```

### Blender (for testing only)

* Download [nerf_synthetic.zip](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract to `data/nerf_synthetic`.

### Real Forward Facing (for testing only)

* Download [nerf_llff_data.zip](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract to `data/nerf_llff_data`.

## Testing

### MVSNeRF Setting (3 Nearest Views)

Download the pretrained model [matchnerf_3v.pth](https://drive.google.com/file/d/1Powy38EOtsrMYN7nh5rx5ySMJ7LUgGSq) and save to `configs/pretrained_models/matchnerf_3v.pth`, then run

```bash
python test.py --yaml=test --name=matchnerf_3v
```

If encounters CUDA out-of-memory, please reduce the ray sampling number, e.g., append `--nerf.rand_rays_test==4096` to the command.

Performance should be exactly the same as below,

| Dataset    | PSNR  | SSIM  | LPIPS |
| ------- | ------| ----- | ------|
| DTU                  | 26.91 | 0.934 | 0.159 |
| Real Forward Facing  | 22.43 | 0.805 | 0.244 |
| Blender | 23.20 | 0.897 | 0.164 |

## Training

Download the GMFlow pretrained weight ([gmflow_sintel-0c07dcb3.pth](https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view)) from  the original [GMFlow repo](https://github.com/haofeixu/gmflow), and save it to `configs/pretrained_models/gmflow_sintel-0c07dcb3.pth`, then run

```bash
python train.py --yaml=train
```

## Rendering Video

```bash
python test.py --yaml=test_video --name=matchnerf_3v_video
```

Results (without any per-scene fine-tuning) should be similar as below,

<details>
  <summary>Visual Results</summary>

![dtu_scan38_view24](https://i.imgur.com/r2vtiaL.gif)<br>
*DTU: scan38_view24*

![blender_materials_view36](https://i.imgur.com/eMZjC1K.gif)<br>
*Blender: materials_view36*

![llff_leaves_view13](https://i.imgur.com/oLaKtMX.gif)<br>
*Real Forward Facing: leaves_view13*

</details>


## Use Your Own Data

* Download the model ([matchnerf_3v_ibr.pth](https://drive.google.com/file/d/1eGY_pkPxxWiSbGFn-Ype8JvW9GqYVfiq)) pretrained with IBRNet data (follow 'GPNR Setting 1'), and save it to `configs/pretrained_models/matchnerf_3v_ibr.pth`.
* Following the instructions detailed in the [LLFF repo](https://github.com/Fyusion/LLFF#1-recover-camera-poses), use [img2poses.py](https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py) to recover camera poses.
* Update the colmap data loader at `datasets/colmap.py` accordingly.

We provide the following 3 input views demo for your reference.

```bash
# lower resolution but fast
python test.py --yaml=demo_own
# full version
python test.py --yaml=test_video_own
```

The generated video will look like,

![colmap_printer](https://i.imgur.com/3boKX8u.gif)<br>
*Demo: own data, printer*


## Miscellaneous

### Citation

If you use this project for your research, please cite our paper.

```bibtex
@article{chen2023matchnerf,
    title={Explicit Correspondence Matching for Generalizable Neural Radiance Fields},
    author={Chen, Yuedong and Xu, Haofei and Wu, Qianyi and Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei},
    journal={arXiv preprint arXiv:2304.12294},
    year={2023}
}
```

### Pull Request

You are more than welcome to contribute to this project by sending a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### Acknowledgments

This implementation borrowed many code snippets from [GMFlow](https://github.com/haofeixu/gmflow), [MVSNeRF](https://github.com/apchenstu/mvsnerf), [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) and [GIRAFFE](https://github.com/autonomousvision/giraffe). Many thanks for all the above mentioned projects.
