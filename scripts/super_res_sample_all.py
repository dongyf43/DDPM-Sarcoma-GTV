"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    test_data_list = os.listdir(args.base_samples)
    for test_data in test_data_list:
        test_data_dir = os.path.join(args.base_samples, test_data)
        save_folder = test_data.split('.')[0]
        save_dir = os.path.join(args.save_dir, save_folder)

        if os.path.exists(save_dir):
            pass
        else:
            dist_util.setup_dist()
            logger.configure(dir=save_dir)

            logger.log("creating model...")
            model, diffusion = sr_create_model_and_diffusion(
                **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                dist_util.load_state_dict(args.model_path, map_location="cpu")
            )
            model.to(dist_util.dev())
            if args.use_fp16:
                model.convert_to_fp16()
            model.eval()

            logger.log("loading data...")
            data = load_data_for_worker(test_data_dir, args.batch_size, args.class_cond)

            sample_num = np.load(test_data_dir)['arr_0'].shape[0]
            args.num_samples = sample_num

            logger.log("creating samples...")
            all_images = []
            while len(all_images) * args.batch_size < args.num_samples:
                model_kwargs = next(data)
                model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
                sample = diffusion.p_sample_loop(
                    model,
                    (args.batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3]),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(all_samples, sample)  # gather not supported with NCCL
                for sample in all_samples:
                    all_images.append(sample.cpu().numpy())
                logger.log(f"created {len(all_images) * args.batch_size} samples")

            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if dist.get_rank() == 0:
                shape_str = "x".join([str(x) for x in arr.shape])
                out_path = os.path.join(save_dir, f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}.npz")
                logger.log(f"saving to {out_path}")
                np.savez(out_path, arr)

            dist.barrier()
            logger.log("sampling complete")
        

def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        image_arr = image_arr[:,0:3,:,:]  # 0:PET 1:CT 2:MRI
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                # batch = batch / 127.5 - 1.0
                # batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
