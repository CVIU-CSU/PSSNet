# Many Birds, One Stone: General Purpose Medical Image Segmentation with Multiple Partially Labelled Datasets  

## Environment

This code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/)

* python == 3.8

* Pytorch == 1.7.0

* mmcv-full == 1.3.15

* sklearn

* tensorboard

* imgviz

```shell
conda create -n mmseg-0.18 python=3.8 -y
conda activate mmseg-0.18

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
pip install mmcv-full==1.3.15  -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html 

cd ~/mmseg-0.18/mmsegmentation-0.18.0  # Change the path to your path
pip install -v -e .  

pip install sklearn 
pip install tensorboard 
pip install future dataclasses
pip install imgviz
```



## Dataset Preparations

We use [DDR](https://github.com/nkicsl/DDR-dataset), [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [REFUGE](https://refuge.grand-challenge.org/), [RETA](https://reta-benchmark.org/), [ARIA](http://www.damianjjfarnell.com/?page_id=276), [DRIVE](https://drive.grand-challenge.org/), [STARE](http://cecas.clemson.edu/~ahoover/stare/). We crop the field of view region from the fundus image and pad it with zeros so that the short and long sides are of equal length.

Data pre-processing consists of three main steps:

1. generate annotations
2. crop the field of view region
3. pad the image with zeros

Please see [prepare_dataset](prepare_dataset)

 The folder should be structured as:

```none
|mmsegmentation-0.18.0/
|data/
│—— FOVCrop-padding/  
|	|--DDR-FOVCrop-padding
|	|--RETA
|	|--REFUGE-FOVCrop-padding
|	|--DRIVE-FOVCrop-padding
|	|--STARE-FOVCrop-padding
|	|--IDRiD-FOVCrop-padding
|	|--ARIA-FOVCrop-padding
```

## Training

The training process consists of two stages: pseudo label generator training and adversarial retraining with pseudo and partial GT labels. 

1. Train pseudo label generator

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_train_multi_task.sh mmsegmentation-0.18.0/configs/_multi_task_/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge.py 4
   ```

2. Generate pseudo labels. Set test dataset in config file, mode and output_dir, then run

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_task_/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge.py work_dirs/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge/iter_60000.pth 4 --vis --source mode --output_dir output_dir
   ```

   Please note that the pseudo-label path in the [config file](configs/_base_/datasets/multi_task_vessel_lesion_OD_idrid_ddr_refuge_semi_OD.py) should correspond to the actual path.

3. Adversarially retrain the segmentation model with pseudo and partial GT labels

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_train_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py 4
   ```

## Evaluation

Set test dataset in config file, then run

```shell
bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py work_dirs/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16/ckpt/iter_60000.pth 4 --eval mIoU 
```

To evaluate the OD performance of IDRiD dataset, run

```shell
bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py work_dirs/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16/ckpt/iter_60000.pth 4 --eval mIoU  --OD
```

## Models

We provide the final model and training logs [here](https://drive.google.com/file/d/1Egss_17VzDCRli3YD-6V5BCRH0ZrQ-FW/view)

## Citation

