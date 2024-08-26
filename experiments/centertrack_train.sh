# ./centertrack_train.sh
# ./centertrack_train.sh --conv_3d_aggregation
# ./centertrack_train.sh --dcn_3d_aggregation

option=$1
exp_id=gen4_centertrack${option//"--"/"_"}
echo "exp_id = $exp_id"

case "$exp_id" in
      gen4_centertrack)
      model_id="model_47"
      ;;
      gen4_centertrack_conv_3d_aggregation)
      model_id="model_70"
      ;;
      gen4_centertrack_dcn_3d_aggregation)
      model_id="model_78"
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
      $option
cd ..
