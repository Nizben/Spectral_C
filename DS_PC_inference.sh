for i in $(seq -w 0000 1002); do
  echo "START line_${i}" 
  CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml \
    --output_folder ./output/DS_PC/line_${i} \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path ./prompts/MovieGenVideoBench_txt/line_${i}.txt \
    --use_ema 
done