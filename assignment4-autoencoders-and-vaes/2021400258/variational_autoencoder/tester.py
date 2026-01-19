import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# DONT CHANGE ANYTHING HERE
def test(model):
    model.eval()

    labels = torch.tensor([0,0,0,0,0,0,0,0,0,0,
                           1,1,1,1,1,1,1,1,1,1,
                           2,2,2,2,2,2,2,2,2,2,
                           3,3,3,3,3,3,3,3,3,3,
                           4,4,4,4,4,4,4,4,4,4,
                           5,5,5,5,5,5,5,5,5,5,
                           6,6,6,6,6,6,6,6,6,6,
                           7,7,7,7,7,7,7,7,7,7,
                           8,8,8,8,8,8,8,8,8,8,
                           9,9,9,9,9,9,9,9,9,9])
    responses = model.infer(labels)
    grid = make_grid(responses, nrow=10, normalize=False, pad_value=1)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1,2,0).squeeze().clip(0, 1), cmap='gray')
    plt.title("Test Results")
    plt.axis('off')
    plt.show()


def test_with_betas(model, betas=[0.0, 0.5, 1.0], use_learned_stats=True):
    model.eval()

    labels = torch.tensor([0,0,0,0,0,0,0,0,0,0,
                           1,1,1,1,1,1,1,1,1,1,
                           2,2,2,2,2,2,2,2,2,2,
                           3,3,3,3,3,3,3,3,3,3,
                           4,4,4,4,4,4,4,4,4,4,
                           5,5,5,5,5,5,5,5,5,5,
                           6,6,6,6,6,6,6,6,6,6,
                           7,7,7,7,7,7,7,7,7,7,
                           8,8,8,8,8,8,8,8,8,8,
                           9,9,9,9,9,9,9,9,9,9])

    num_betas = len(betas)
    fig, axes = plt.subplots(1, num_betas, figsize=(8 * num_betas, 10))
    
    if num_betas == 1:
        axes = [axes]
    
    mode_str = "Learned Stats" if use_learned_stats else "Random N(0,1)"
    
    for ax, beta in zip(axes, betas):
        responses = model.infer(labels, beta=beta, use_learned_stats=use_learned_stats)
        responses_cpu = responses.detach().cpu()
        grid = make_grid(responses_cpu, nrow=10, normalize=False, pad_value=1)
        
        ax.imshow(grid[0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"β = {beta}", fontsize=14)
        ax.axis('off')
    
    plt.suptitle(f"VAE Inference Results ({mode_str}) with Different β Values", fontsize=16)
    plt.tight_layout()
    plt.show()


def test_compare_modes(model, beta=0.4):
    model.eval()

    labels = torch.tensor([0,0,0,0,0,0,0,0,0,0,
                           1,1,1,1,1,1,1,1,1,1,
                           2,2,2,2,2,2,2,2,2,2,
                           3,3,3,3,3,3,3,3,3,3,
                           4,4,4,4,4,4,4,4,4,4,
                           5,5,5,5,5,5,5,5,5,5,
                           6,6,6,6,6,6,6,6,6,6,
                           7,7,7,7,7,7,7,7,7,7,
                           8,8,8,8,8,8,8,8,8,8,
                           9,9,9,9,9,9,9,9,9,9])

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Learned stats mode
    responses_learned = model.infer(labels, beta=beta, use_learned_stats=True)
    responses_cpu = responses_learned.detach().cpu()
    grid_learned = make_grid(responses_cpu, nrow=10, normalize=False, pad_value=1)
    
    axes[0].imshow(grid_learned[0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Learned Stats (β={beta})", fontsize=14)
    axes[0].axis('off')
    
    # Random sampling mode (original VAE behavior)
    responses_random = model.infer(labels, use_learned_stats=False)
    responses_cpu = responses_random.detach().cpu()
    grid_random = make_grid(responses_cpu, nrow=10, normalize=False, pad_value=1)
    
    axes[1].imshow(grid_random[0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Random N(0,1) Sampling", fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle("VAE Inference: Learned Stats vs Random Sampling", fontsize=16)
    plt.tight_layout()
    plt.show()
