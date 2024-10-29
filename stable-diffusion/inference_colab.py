import argparse
import os
import sys
import json
import torch
from torch import autocast
from contextlib import contextmanager, nullcontext
from tqdm import tqdm
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_lightning import seed_everything
import time
import torch.nn as nn

from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler


def load_prompts_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")

    # Load checkpoint weights
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]

    # Instantiate model from config
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)

    # Get token embedding sizes
    model_token_embedding_size = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.size(
        0
    )
    ckpt_token_embedding_size = sd[
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
    ].size(0)

    # If sizes don't match, adjust the checkpoint's token embedding weights
    if model_token_embedding_size != ckpt_token_embedding_size:
        print(
            f"Adjusting token embedding size from {ckpt_token_embedding_size} to {model_token_embedding_size}"
        )
        old_embedding = sd[
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
        ]
        new_embedding = torch.nn.Parameter(
            torch.zeros(model_token_embedding_size, old_embedding.size(1))
        )
        # Copy existing embeddings
        new_embedding.data[:ckpt_token_embedding_size, :] = old_embedding
        # Initialize new embeddings (if model size is larger)
        if model_token_embedding_size > ckpt_token_embedding_size:
            nn.init.normal_(new_embedding.data[ckpt_token_embedding_size:, :])
        # Update checkpoint
        sd[
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
        ] = new_embedding

    # Load state dict
    m, u = model.load_state_dict(sd, strict=False)

    # Print missing or unexpected keys (if verbose is True)
    if verbose:
        if len(m) > 0:
            print("Missing keys:")
            print(m)
        if len(u) > 0:
            print("Unexpected keys:")
            print(u)

    return model


# def load_model_from_config(config, ckpt, verbose=False):
#     print(f"Loading model from {ckpt}")

#     # 加载检查点权重
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     sd = pl_sd["state_dict"]

#     # 从配置实例化模型
#     config.model.params.ckpt_path = ckpt
#     model = instantiate_from_config(config.model)

#     # 获取模型中的词嵌入层大小
#     model_token_embedding_size = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.size(
#         0
#     )

#     # 获取检查点中的词嵌入层大小
#     ckpt_token_embedding_size = sd[
#         "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
#     ].size(0)

#     # 如果大小不匹配，跳过加载 token_embedding 的权重
#     if model_token_embedding_size != ckpt_token_embedding_size:
#         print(
#             f"Skipping loading of token_embedding due to size mismatch: {ckpt_token_embedding_size} (ckpt) vs {model_token_embedding_size} (model)"
#         )
#         sd.pop(
#             "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
#         )

#     # 加载剩余的模型权重
#     m, u = model.load_state_dict(sd, strict=False)

#     # 打印丢失或多余的权重（如果 verbose 为 True）
#     if verbose:
#         if len(m) > 0:
#             print("Missing keys:")
#             print(m)
#         if len(u) > 0:
#             print("Unexpected keys:")
#             print(u)

#     return model


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="number of ddim sampling steps"
    )
    parser.add_argument(
        "--seed", type=int, default=444, help="the seed (for reproducible sampling)"
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument("--n_iter", type=int, default=25, help="sample this often")
    parser.add_argument(
        "--H", type=int, default=512, help="image height, in pixel space"
    )
    parser.add_argument(
        "--W", type=int, default=512, help="image width, in pixel space"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument(
        "--scale", type=float, default=9.4, help="unconditional guidance scale"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    # parser.add_argument("--embedding_path", type=str, help="path to the embedding file")
    opt = parser.parse_args()
    seed_everything(opt.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # json_path = sys.argv[1]
    # output_path = sys.argv[2]
    # ckpt_path = sys.argv[3]
    # json_path = "./hw2_data/textual_inversion/input.json"
    json_path = sys.argv[1]
    # output_path = "./outputs_2"
    output_path = sys.argv[2]
    # ckpt_path = "./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
    ckpt_path = sys.argv[3]
    config_file = "./stable-diffusion/configs/stable-diffusion/inf.yaml"
    # config_file_1 = "/home/jay/Project/dlcv-fall-2024-hw2-JayYang920110/2-3/stable-diffusion/configs/stable-diffusion/1.yaml"
    config = OmegaConf.load(config_file)  # 4800 5400
    embedding_path_0 = "./embeddings_gs-1399.pt"
    embedding_path_1 = "./embeddings_gs-1494.pt"

    data = load_prompts_from_json(json_path)
    for key, text in data.items():
        print(f"key : {key}")
        src_image = text["src_image"]
        token_name = text["token_name"]
        prompts = text["prompt"]

        if key == "0":
            # continue
            opt.scale = 8.15
            config.model.params.personalization_config.params.placeholder_strings = [
                token_name
            ]

            config.model.params.personalization_config.params.num_vectors_per_token = 2
            config.model.params.cond_stage_config.params.new_tokens = "<new1>"
            model = load_model_from_config(config, ckpt_path)
            model.embedding_manager.load(embedding_path_0)

        if key == "1":
            # opt.scale = 5
            config.model.params.personalization_config.params.placeholder_strings = [
                token_name
            ]
            config.model.params.personalization_config.params.num_vectors_per_token = 3
            config.model.params.cond_stage_config.params.new_tokens = token_name
            model = load_model_from_config(config, ckpt_path)
            model.embedding_manager.load(embedding_path_1)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

        output_dir = os.path.join(output_path, key)
        os.makedirs(output_dir, exist_ok=True)

        # sampler = DDIMSampler(model)
        sampler = DPMSolverSampler(model)
        # sampler = PLMSSampler(model)

        pbar = tqdm(enumerate(prompts), desc="Processing prompts")
        for prompt_index, prompt in pbar:
            print(f"prompt : {prompt}")

            prompt_dir = os.path.join(output_dir, str(prompt_index))
            os.makedirs(prompt_dir, exist_ok=True)

            base_count = 0  # 記錄已生成的圖片數量
            all_samples = []
            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.n_samples * [""])
                        for n in tqdm(range(opt.n_iter), desc="Sampling"):
                            # 如果已生成24張圖片且是最後一次迭代，設定為只保存一張圖片
                            if base_count >= 24 and n == opt.n_iter - 1:
                                opt.n_samples = 1  # 將生成圖片數量限制為1張

                            # 若已生成的圖片數量達到25張，則退出迴圈
                            if base_count >= 25:
                                break

                            c = model.get_learned_conditioning(opt.n_samples * [prompt])
                            shape = [4, opt.H // 8, opt.W // 8]
                            samples_ddim, _ = sampler.sample(
                                S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )

                            images = []
                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                images.append(
                                    Image.fromarray(x_sample.astype(np.uint8))
                                )

                            # 批量保存圖片，並確保不超過25張
                            for i, img in enumerate(images):
                                if base_count >= 25:
                                    break
                                img.save(
                                    os.path.join(prompt_dir, f"{base_count:04}.png")
                                )
                                base_count += 1

                            all_samples.append(x_samples_ddim)

                            # 恢復 opt.n_samples 以便繼續下次迭代
                            if n == opt.n_iter - 1 and base_count < 25:
                                opt.n_samples = 2
    end = time.time()
    print(f"Total {end-start} seconds")


if __name__ == "__main__":
    main()
