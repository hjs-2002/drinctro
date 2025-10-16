# default
MODEL_NAME=clip-ViT-L-14
# MODEL_NAME=convnext_base_in22k
# MODEL_PATH=pretrained/GenImage/sdv14/convnext_base_in22k_224_drct_amp_crop/last_acc0.9991.pth
# MODEL_PATH=pretrained/GenImage/sdv14/clip-ViT-L-14_224_drct_amp_crop/2_acc0.9558.pth
# MODEL_PATH=output/GenImage/1/convnext_base_in22k_224_drct_amp_crop/weights/0_acc0.9962.pth
# MODEL_PATH=output/GenImage/1/clip-ViT-L-14_224_drct_amp_crop/weights/last_acc0.9667.pth
# MODEL_PATH=output/GenImage/2/convnext_base_in22k_224_drct_amp_crop/weights/2_acc0.9980.pth
# MODEL_PATH=output/GenImage/2/clip-ViT-L-14_224_drct_amp_crop/weights/6_acc0.9671.pth
# MODEL_PATH=output_more_epoch/GenImage/1/clip-ViT-L-14_224_drct_amp_crop/weights/last_acc0.9680.pth
# MODEL_PATH=pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth
MODEL_PATH=pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth

DEVICE_ID=3
EMBEDDING_SIZE=1024
MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${2:-$MODEL_PATH}
DEVICE_ID=${3:-$DEVICE_ID}
EMBEDDING_SIZE=${4:-$EMBEDDING_SIZE}
ROOT_PATH=/home/law/data/GenImage
FAKE_ROOT_PATH=""
DATASET_NAME=GenImage
SAVE_TXT=./output_more_epoch/results/GenImage_metrics_rebuttal.txt
INPUT_SIZE=224
BATCH_SIZE=24
FAKE_INDEXES=(1 2 3 4 5 6 7 8)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
# do
#   echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
#   python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
#                   --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
#                   --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
#                   --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
# done

# do
#   echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
#   nohup python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
#                   --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
#                   --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
#                   --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE} > output_GenImage_test_convB.log 2>&1
# done

#在univFD上测试
# do
#   echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
#   nohup python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
#                   --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
#                   --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
#                   --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}> output_GenImage_test_UnivFD.log 2>&1 
# done

do
  echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
  python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done

# bash test_GenImage.sh clip-ViT-L-14 pretrained/GenImage/sdv14/clip-ViT-L-14_224_drct_amp_crop/2_acc0.9558.pth