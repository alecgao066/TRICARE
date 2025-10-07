python create_splits_seq.py --seed 3 --k 8  --leave_one_out 

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --lr 2e-4 --k 8 --leave_one_out --agg_range 3 --agg_gap 1 --adj_gap 5 --exp_code exp_prostate_range60gap60 --weighted_sample --max_epochs 5 --bag_loss ce --model_type carp3d_naive --log_data --data_root_dir ../../../../media/cfxuser/new/test_conch_normpatch_prostate2-5_coloreda/ 