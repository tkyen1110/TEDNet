# ./permatrack_train.sh
# ./permatrack_train.sh --consistency
# ./permatrack_train.sh --conv_3d_aggregation
# ./permatrack_train.sh --dcn_3d_aggregation
# ./permatrack_train.sh --conv_3d_aggregation --consistency
# ./permatrack_train.sh --dcn_3d_aggregation --consistency

option1=$1
option2=$2
if [[ "$option1" == "--consistency" ]];
then
      option1=""
      option2=$1
fi
exp_id=gen4_supinvis_permatrack${option1//"--"/"_"}${option2//"--"/"_w_"}
echo "exp_id = $exp_id"

case "$exp_id" in
      gen4_supinvis_permatrack)
      model_id="model_54"
      ;;
      gen4_supinvis_permatrack_w_consistency)
      model_id="model_85"
      ;;
      gen4_supinvis_permatrack_conv_3d_aggregation)
      model_id="model_78"
      ;;
      gen4_supinvis_permatrack_dcn_3d_aggregation)
      model_id="model_71"
      ;;
      gen4_supinvis_permatrack_conv_3d_aggregation_w_consistency)
      model_id="model_102"
      ;;
      gen4_supinvis_permatrack_dcn_3d_aggregation_w_consistency)
      model_id="model_96"
      ;;
      *)
      exit
esac

cd ../src
python main.py tracking \
      --exp_id $exp_id --occlusion_thresh 0.15 --visibility_thresh 0.05 \
      --dataset gen4_tracking --dataset_version train --same_aug_pre --hm_disturb 0.0 \
      --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1 --batch_size 9 \
      --load_model ../exp/tracking/$exp_id/$model_id.pth --val_intervals 200 \
      --is_recurrent --gru_filter_size 7 --input_len 4 --pre_thresh 0.4 \
      --hm_weight 0.5 --num_epochs 200 --lr_step 7 --sup_invis \
      --invis_hm_weight 20 --use_occl_len --occl_len_mult 5 --num_iter -1 \
      --no_color_aug --const_v_2d --num_workers 0 --annotations_dir annotations \
      --visibility --visibility_thresh_eval 0.2 \
      $option1 $option2
cd ..
