from torchvision import datasets, transforms
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader


def train(model, data_dir, epoch_count, save_path, save_interval=-1, kl_beta=0.001):
    # You may change the input parameters but don't change the data_directory and the model save path

    # You can change the normalization transformation defined here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # Maps [0,1] to [-1,1] to match tanh output range
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
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epoch_count):
        epoch_loss = 0
        epoch_iteration = 0

        for batch_index, (data, labels) in enumerate(train_loader):
            # implement a training step
            # reset your optimizer
            # get reconstructed image, mean and std values

            data = data.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()

            out_data, mu, l = model(data, labels)

            kl_loss = -0.5 * torch.sum(1+l-mu.pow(2)-l.exp())

            loss = loss_fn(out_data, data) + kl_loss * kl_beta

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_iteration += 1
            if batch_index % 100 == 0:
                print(f'{epoch}-{batch_index}\tbatch loss:\t', loss.item())

        epoch_loss /= epoch_iteration
        print(f'Epoch {epoch} loss: {epoch_loss:.4f}')

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            model.save(save_path)

    model.eval()
    data_indices = []
    for i in range(10):
        data_indices.append(torch.nonzero(train_dataset.targets == i).flatten()[:10])
    data_indices = torch.cat(data_indices, 0)
    subset_dataset = Subset(train_dataset, data_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=100, shuffle=False)

    data, labels = next(iter(subset_loader))
    data = data.to(model.device)
    labels = labels.to(model.device)
    out_data, _, _ = model(data, labels)
    grid = make_grid(out_data.cpu(), nrow=10, normalize=False, pad_value=1)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze().clip(0, 1), cmap='gray')
    plt.title("Variational Auto Encoder Train Results")
    plt.axis('off')
    plt.show()

    # recording and saving distribution may not be necessary
    # model already uses mu and l
    # for data, labels in train_loader:
    #     model.record_distribution(data, labels)
    # model.save_distribution()
    model.save(save_path)
    print('model saved to', save_path)
