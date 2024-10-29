python3 main.py \
    --actual_resume /home/chrishsieh/dev/DLCV/textual_inversion/models/ldm/stable-diffusion-v1/model.ckpt \
    --data_root /home/chrishsieh/dev/DLCV/textual_inversion/hw2_data/textual_inversion/1 \
    --placeholder_string "<new2>" \
    --learnable_property "object" \
    --train \
    --seed 42 \
    --embedding_manager_ckpt /home/chrishsieh/dev/DLCV/textual_inversion/embeddings_gs-1003.pt \
    --verbose
