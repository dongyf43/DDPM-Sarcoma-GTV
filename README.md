# DDPM-Sarcoma-GTV
This is the official repository for article "Yafei Dong, Thibault Marin, Yue Zhuo, Elie Najem, Maryam Moteabbed, Fangxu Xing, Arnaud Beddok, Rita Maria Lahoud, Laura Rozenblum, Zhiyuan Ding, Xiaofeng Liu, Kira Grogg, Jonghye Woo, Yen-Lin E. Chen, Ruth Lim, Chao Ma, Georges El Fakhri, Gross tumor volume confidence maps prediction for soft tissue sarcomas from multi-modality medical images using a diffusion model, Physics and Imaging in Radiation Oncology, 2025, 100734".

## Data Preparation
To train the model, please firstly modify `class ImageDataset(Dataset)` in `/guided_diffusion/image_datasets.py` according to your dataset. In our dataset, we stored training data in `.npz` files containing 2D transverse slices with sizes of `3x512x512`, where `3` is the number of channels (0: PET, 1: CT, 2: MRI). Similarly, to do inferene, please modify `def load_data_for_worker(base_samples, batch_size, class_cond)` in `/scripts/super_res_sample_all.py`. We stored each testing case in one `.npz` file with size of `Nx3x512x512`, where `N` is the number of slices for each case.

## Model Training
To train the model, please use:
```
TRAIN_FLAGS="--batch_size 8 --save_interval 50000"
MODEL_FLAGS="--attention_resolutions 1 --class_cond False --diffusion_steps 1000 --dropout 0.1 --large_size 128 --small_size 128 --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --learn_sigma True --noise_schedule linear --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python /scripts/super_res_train.py --data_dir /xxx/xxx $MODEL_FLAGS $TRAIN_FLAGS
```

## Model Inference
To do inference, please use:
```
SAMPLE_FLAGS="--batch_size 1 --num_samples 144"
MODEL_FLAGS="--attention_resolutions 1 --class_cond False --diffusion_steps 1000 --dropout 0.1 --large_size 128 --small_size 128 --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --learn_sigma True --noise_schedule linear --resblock_updown False --use_fp16 True --use_scale_shift_norm True"
python /scripts/super_res_sample_all.py $MODEL_FLAGS --model_path xxx/best.pt --save_dir xxx/xxx --base_samples xxx/xxx $SAMPLE_FLAGS
```
The code could do inference for all cases in the data folder `--base_samples xxx/xxx` in one time.

## Evaluation
We also provide the code to evaluate the performance according to metrics we used in the manuscript. Please refer to `/evaluation/evaluation_all_ddpm.py`.

If you find this helpful, please consider to cite our work:
```
@article{DONG2025100734,
title = {Gross tumor volume confidence maps prediction for soft tissue sarcomas from multi-modality medical images using a diffusion model},
journal = {Physics and Imaging in Radiation Oncology},
volume = {33},
pages = {100734},
year = {2025},
issn = {2405-6316},
doi = {https://doi.org/10.1016/j.phro.2025.100734},
url = {https://www.sciencedirect.com/science/article/pii/S2405631625000399},
author = {Yafei Dong and Thibault Marin and Yue Zhuo and Elie Najem and Maryam Moteabbed and Fangxu Xing and Arnaud Beddok and Rita Maria Lahoud and Laura Rozenblum and Zhiyuan Ding and Xiaofeng Liu and Kira Grogg and Jonghye Woo and Yen-Lin E. Chen and Ruth Lim and Chao Ma and Georges {El Fakhri}},
keywords = {Sarcoma, Primary gross tumor volume, Confidence map, Deep learning, Diffusion model},
}
```
