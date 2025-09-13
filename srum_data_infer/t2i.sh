# #!/bin/bash

python  t2i_multi_gpu.py \
    --model_path "BAGEL-7B-MoT" \
    --output_dir images_wise/bagel_base_7b_50_step_think \
    --metadata_file wise.json\
    --num_timesteps "50"\
    --num_gpus 8\
    --think \

    