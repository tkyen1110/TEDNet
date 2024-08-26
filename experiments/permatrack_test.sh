# ./permatrack_test.sh [model_id] 
# ./permatrack_test.sh [model_id] --consistency
# ./permatrack_test.sh [model_id] --conv_3d_aggregation
# ./permatrack_test.sh [model_id] --dcn_3d_aggregation
# ./permatrack_test.sh [model_id] --conv_3d_aggregation --consistency
# ./permatrack_test.sh [model_id] --dcn_3d_aggregation --consistency

model_id=$1
option1=$2
option2=$3
if [[ "$option1" == "--consistency" ]];
then
      option1=""
      option2=$2
fi
exp_id=gen4_supinvis_permatrack${option1//"--"/"_"}${option2//"--"/"_w_"}
echo "exp_id = $exp_id"

cd ../src
python test.py tracking \
      --exp_id $exp_id \
      --dataset gen4_tracking --dataset_version test --track_thresh 0.1 \
      --load_model ../exp/tracking/$exp_id/$model_id.pth \
      --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --stream_test \
      --debug 4 --gpus 0 --input_len 1 --batch_size 4 --flip 0.0 \
      --visibility --visibility_thresh_eval 0.4 --annotations_dir annotations --save_npys \
      $option1
cd ..
