from torchvision import datasets, transforms
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader


def train(model, data_dir, epoch_count, save_path, save_interval=-1, sample_percentage=0.2):
    # You may change the input parameters but don't change the data_directory and the model save path
    # sample_percentage: Fraction of samples per class to compute latent statistics (e.g., 0.1 for 10%)
    # beta: Scaling factor for std deviation in latent sampling during inference


    # You can change the normalization transformation defined here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model.train()

    # define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # define your loss function to calculate reconstruction loss
    criterion = torch.nn.MSELoss()

    for epoch in range(epoch_count):
        epoch_loss = 0
        epoch_iteration = 0

        for batch_index, (data, labels) in enumerate(train_loader):
            # implement a training step
            # reset your optimizer
            # take prediction from your model
            # calculate loss
            # and backpropagate

            data = data.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()

            outputs = model.forward(data, labels)

            # loss between reconstructed images and the original data
            loss = criterion(outputs, data)

            # backwards pass
            loss.backward()
            optimizer.step()

            # This part handles local logging, you may add other
            # metrics, but don't remove the existing ones
            # loss = torch.zeros(1)
            epoch_loss += loss.item()
            epoch_iteration += 1
            if batch_index % 100 == 0:
                print(f'{epoch}-{batch_index}\tbatch loss:\t', loss.item())
        epoch_loss /= epoch_iteration
        print(f'Epoch {epoch} loss: {epoch_loss:.4f}')

        # This part may work as an insurance,
        #   if your training gets interrupted,
        #   you may continue your training from
        #   the last checkpoint.
        if save_interval > 0 and (epoch+1)%save_interval == 0:
            model.save(save_path)

    model.eval()

    # Compute per-class latent statistics for inference
    model.compute_latent_stats(train_dataset, sample_percentage)

    data_indices = []
    for i in range(10):
        data_indices.append(torch.nonzero(train_dataset.targets==i).flatten()[:10])
    data_indices = torch.cat(data_indices, 0)
    subset_dataset = Subset(train_dataset, data_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=100, shuffle=False)

    data, labels = next(iter(subset_loader))
    data = data.to(model.device)
    labels = labels.to(model.device)
    out_data = model(data, labels)
    grid = make_grid(out_data.cpu(), nrow=10, normalize=False, pad_value=1)


    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1,2,0).squeeze().clip(0, 1), cmap='gray')
    plt.title("Train Results")
    plt.axis('off')
    plt.show()

    # Save the final version of the trained model
    model.save(save_path)
    print('model saved to', save_path)
