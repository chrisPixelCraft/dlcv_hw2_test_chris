python scripts/txt2img.py \
    --ddim_eta 0.0 \
    --n_samples 8 \
    --n_iter 2 \
    --scale 10.0 \
    --ddim_steps 50 \
    --embedding_path embeddings_gs-6099.pt \
    --ckpt_path models/ldm/stable-diffusion-v1/model.ckpt \
    --prompt "The streets of Paris in the style of <new2>." \
    --outdir outputs
