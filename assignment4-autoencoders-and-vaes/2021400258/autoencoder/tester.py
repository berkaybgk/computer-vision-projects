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
    
    # Debug output
    print(f"Test infer shape: {responses.shape}, range: [{responses.min():.3f}, {responses.max():.3f}]")
    
    responses_cpu = responses.detach().cpu()
    grid = make_grid(responses_cpu, nrow=10, normalize=False, pad_value=1)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid[0].numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title("Test Results")
    plt.axis('off')
    plt.show()


def test_with_betas(model, betas=[0.0, 0.5, 1.0]):
    # Use beta values to generate examples with more variance
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
    
    for ax, beta in zip(axes, betas):
        responses = model.infer(labels, beta=beta)
        responses_cpu = responses.detach().cpu()
        grid = make_grid(responses_cpu, nrow=10, normalize=False, pad_value=1)
        
        ax.imshow(grid[0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"β = {beta}", fontsize=14)
        ax.axis('off')
    
    plt.suptitle("Inference Results with Different β Values", fontsize=16)
    plt.tight_layout()
    plt.show()
