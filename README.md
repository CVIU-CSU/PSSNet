# Many Birds, One Stone: General Purpose Medical Image Segmentation with Multiple Partially Labelled Datasets  

## Environment

This code us based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/)

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

We Use [DDR](https://github.com/nkicsl/DDR-dataset), [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [REFUGE](https://refuge.grand-challenge.org/), [RETA](https://reta-benchmark.org/), [ARIA](http://www.damianjjfarnell.com/?page_id=276), [DRIVE](https://drive.grand-challenge.org/), [STARE](http://cecas.clemson.edu/~ahoover/stare/). We crop black image background and pad the image with value zero so that the short and long sides are the same length. 

Data pre-processing consists of three main steps:

1. generate annotations
2. crop black image background
3. pad the image with value zero

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

1. Pseudo label generator training

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_train_multi_task.sh mmsegmentation-0.18.0/configs/_multi_task_/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge.py 4
   ```

2. To generate pseudo label, set test dataset in config file, mode and output_dir, run

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_task_/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge.py work_dirs/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_group_idrid_ddr_refuge/iter_60000.pth 4 --vis --source mode --output_dir output_dir
   ```

3. Place the generated pseudo label in the appropriate directory. With the DDR dataset as an example,  The folder should be structured as

   ```none
   |data/
   │—— FOVCrop-padding/
   │   ├—— DDR-FOVCrop-padding/  
   |   |   ├—— train/
   |   |       ├—— ddr_ODOC_pseudo/
   |   |       ├—— ddr_vessel_pseudo/
   ```

4. Adversarial retraining with pseudo and partial GT labels

   ```shell
   bash mmsegmentation-0.18.0/tools/dist_train_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py 4
   ```

## Evaluation

Set test dataset in config file, run

```shell
bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py work_dirs/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16/ckpt/iter_60000.pth 4 --eval mIoU 
```

To evaluate the OD performance of IDRiD dataset, run

```shell
bash mmsegmentation-0.18.0/tools/dist_test_gan.sh mmsegmentation-0.18.0/configs/_multi_gan_/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16.py work_dirs/gan_bdice_vessel_lesion_OD_idrid_ddr_refuge_semi_OD_0.1_0.05_60k_128_group_window_size16/ckpt/iter_60000.pth 4 --eval mIoU  --OD
```

## Models

We provide the final model and training logs [here](https://pan.baidu.com/s/1g7wKibZQd9y5XsKPGHhE1w?pwd=or7n)

## Citation

