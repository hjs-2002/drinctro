# default
# MODEL_NAME=convnext_base_in22k
MODEL_NAME=clip-ViT-L-14
# MODEL_PATH=pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth
# MODEL_PATH=output/DRCT-2M/2/convnext_base_in22k_224_drct_amp_crop/weights/7_acc0.9999.pth
# MODEL_PATH=pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth
# MODEL_PATH=output/DRCT-2M/2/convnext_base_in22k_224_drct_amp_crop/weights/0_acc0.9951.pth
# MODEL_PATH=output/DRCT-2M/2/clip-ViT-L-14_224_drct_amp_crop/weights/2_acc0.9721.pth
# MODEL_PATH=output/DRCT-2M/2/convnext_base_in22k_224_drct_amp_crop/weights/7_acc0.9999.pth
# MODEL_PATH=output/DRCT-2M/2/convnext_base_in22k_224_drct_amp_crop/weights/0_acc0.9921.pth
# MODEL_PATH=output_more_epoch/DRCT-2M/2/clip-ViT-L-14_224_drct_amp_crop/weights/last_acc0.9658.pth
MODEL_PATH=output_more_epoch/DRCT-2M/2/clip-ViT-L-14_224_drct_amp_crop/weights/13_acc0.9668.pth

DEVICE_ID=1
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=dataset/MSCOCO
FAKE_ROOT_PATH=dataset/images
DATASET_NAME=DRCT-2M
SAVE_TXT=./output_more_epoch/results/DRCT-2M_metrics.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
# do
#   echo FAKE_INDEX:${FAKE_INDEX}
#   nohup python train.py --root_path ${ROOT_PATH} --fake_root_path ${FAKE_ROOT_PATH} --model_name ${MODEL_NAME} \
#                   --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
#                   --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
#                   --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE} > output_DRCT_test.log 2>&1
# done


do
  echo FAKE_INDEX:${FAKE_INDEX}
  python train.py --root_path ${ROOT_PATH} --fake_root_path ${FAKE_ROOT_PATH} --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}  
done