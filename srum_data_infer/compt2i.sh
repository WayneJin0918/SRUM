# #!/bin/bash

# Think mode
python  t2i_multi_gpu.py \
    --model_path "BAGEL-7B-MoT" \
    --output_dir images_comp/bagel_base_7b_50_step_think \
    --metadata_file train_comp.json\
    --num_timesteps "50"\
    --num_gpus 8\
    --think \

# Normal mode
python  t2i_multi_gpu.py \
    --model_path "BAGEL-7B-MoT" \
    --output_dir images_comp/bagel_base_7b_50_step_think \
    --metadata_file train_comp.json\
    --num_timesteps "50"\
    --num_gpus 8\