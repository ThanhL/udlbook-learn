import argparse
import time
import math
import torch
import matplotlib.pyplot as plt
from model import UNetDiffSystem


class UNetMNISTSampler:
    def __init__(self, model_path):
        self.model_path = model_path

        # -- Load model and put in evaluation
        self.model = UNetDiffSystem.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()

    def sample(self):
        # Initial noisy image
        nimage = torch.randn((1, 1, 28, 28)).to(self.model.device)

        # init scheduler
        if self.model.noise_scheduler.num_inference_steps is None:
            num_inference_steps = \
                self.model.noise_scheduler.config.num_train_timesteps
        else:
            num_inference_steps = \
                self.model.noise_scheduler.num_inference_steps

        self.model.noise_scheduler.set_timesteps(
            num_inference_steps)

        # Diffusion sampling
        for k in self.model.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.model(
                sample=nimage,
                timestep=k,
            )

            # inverse diffusion step (remove noise)
            nimage = self.model.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=nimage
            ).prev_sample

        return nimage


def main():
    # -- Argparser
    parser = argparse.ArgumentParser(
        description="Diffusion MNIST inference.")
    parser.add_argument(
        "--model_path", type=str,
        required=True,
        help="Path to the trained model checkpoint.")
    parser.add_argument(
        "--num_samples", type=int,
        default=20,
        help="Number of samples to generate.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Whether to print verbose logs.")

    args = parser.parse_args()
    assert args.num_samples > 0, "Number of samples must be positive."

    # -- Model init
    unet_mnist_sampler = UNetMNISTSampler(args.model_path)

    # -- Sample 20 images
    fig, axes = plt.subplots(math.ceil(args.num_samples / 5),
                             5, figsize=(12, 10))
    axes = axes.flatten()

    start_time = time.time()
    for i in range(args.num_samples):
        nimage = unet_mnist_sampler.sample()
        x_sample = torch.permute(nimage[0], (1, 2, 0)).detach().cpu().numpy()

        axes[i].imshow(x_sample, cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')

        if args.verbose:
            print(f"[INFO]: Sample {i}, Elasped time: "
                  f"{time.time() - start_time:.4f} seconds")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
