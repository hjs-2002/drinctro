# DRCT-2M训练
nohup python train_test_DRCT_plot.py --root_path /home/data1/Datasets/Dataset_hjs/dataset_DRCT/MSCOCO/train2017,/home/data1/Datasets/Dataset_hjs/dataset_DRCT/images/stable-diffusion-inpainting/train2017 \
                            --fake_root_path /home/data1/Datasets/Dataset_hjs/dataset_DRCT/images/stable-diffusion-v1-4/train2017,/home/data1/Datasets/Dataset_hjs/dataset_DRCT/fake_rec_images/stable-diffusion-v1-4/train2017 \
                            --dataset_name DRCT-2M \
                            --model_name InCTRL \
                            --embedding_size 1024 \
                            --input_size 224 \
                            --batch_size 256 \
                            --fake_indexes 2 \
                            --num_epochs 10 \
                            --device_id 3 \
                            --lr 0.0001 \  --is_crop\
                            --num_workers 12 \
                            --save_flag \
                            --is_amp  _drct_amp_crop > output_DR-inctrl_transform-drct_dire_final_demo.log 2>&1



# # GenImage训练
# nohup python train_contrastive.py --root_path /home/law/data/DR/GenImage \
#                             --fake_root_path /home/law/data/DR/GenImage\
#                             --dataset_name GenImage \
#                             --model_name InCTRL \
#                             --embedding_size 1024 \
#                             --input_size 224 \
#                             --batch_size 256 \
#                             --fake_indexes 1 \
#                             --num_epochs 10 \
#                             --device_id 2\
#                             --lr 0.0001 \
#                             --is_amp \
#                             --is_crop \
#                             --num_workers 12 \
#                             --inpainting_dir inpainting \
#                             --save_flag _drct_amp_crop > output_DR-inctrl_transform-drct_dire_Genimage_all.log 2>&1
