import torch.nn as nn
import torch


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()

        # You can change the network structure
        # You may add or remove any elements
        #   other than the save, load, forward,
        #   and infer functions itself

        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 512), # Added 10 for one-hot encoded class labels
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(256, 128)
        self.logvar_head = nn.Linear(256, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128 + 10, 256), # Added 10 for one-hot encoded class labels
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

        # Register buffers for per-class latent statistics
        self.register_buffer('latent_mean', torch.zeros(10, 128))
        self.register_buffer('latent_std', torch.ones(10, 128))

        self.device = device
        self.to(device)

    def save(self, out_path):
        # You can change the saving methodology, but be careful, you might need to change the load function too
        torch.save({
            "state_dict": self.state_dict(),
        }, out_path)

    @staticmethod
    def load(in_path, device=torch.device('cpu')):
        # You can change the loading methodology, but be careful, you might need to change the save function too
        model_data = torch.load(in_path, map_location=device)
        model = ConditionalVariationalAutoEncoder()
        model.load_state_dict(model_data["state_dict"])
        return model

    def forward(self, x, classes):
        # This forward operation only learns how to encode and decode an input digit image
        # You should apply some changes to enable model to learn how to encode and decode images regarding their classes

        # Concatenate one-hot encoded class labels to the input images, to make it conditional
        input_tensor = x.view(-1, 28*28)
        y = self._one_hot_encode_labels(classes)
        input_tensor = torch.cat((input_tensor, y), dim=1)

        z = self.encoder(input_tensor)
        mu = self.mu_head(z)
        l = self.logvar_head(z)

        # this operation is called reparameterization trick
        std = torch.exp(0.5 * l)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z = torch.cat((z, y), dim=1)

        # generate the reconstructed image using the reparameterized latent vector
        return self.decoder(z).reshape(x.shape), mu, l

    # You can implement other functionalities too as needed

    def compute_latent_stats(self, train_dataset, sample_percentage=0.1):
        self.eval()

        all_means = torch.zeros(10, 128)
        all_stds = torch.ones(10, 128)
        
        with torch.no_grad():
            for class_idx in range(10):
                # Get indices of all samples for this class
                class_indices = torch.nonzero(train_dataset.targets == class_idx).flatten()
                
                # Sample a percentage of them
                num_samples = max(1, int(len(class_indices) * sample_percentage))
                perm = torch.randperm(len(class_indices))[:num_samples]
                selected_indices = class_indices[perm].tolist()
                
                # Collect latent mu vectors for this class
                latent_vectors = []
                for idx in selected_indices:
                    image, label = train_dataset[idx]
                    image = image.unsqueeze(0).to(self.device)
                    label_tensor = torch.tensor([label], device=self.device)
                    
                    # Encode to get mu (mean of latent distribution)
                    input_tensor = image.view(-1, 28*28)
                    y = self._one_hot_encode_labels(label_tensor)
                    input_tensor = torch.cat((input_tensor, y), dim=1)
                    z = self.encoder(input_tensor)
                    mu = self.mu_head(z)
                    latent_vectors.append(mu.detach().cpu())
                
                # Stack and compute statistics
                latent_vectors = torch.cat(latent_vectors, dim=0)
                all_means[class_idx] = latent_vectors.mean(dim=0)
                all_stds[class_idx] = latent_vectors.std(dim=0, unbiased=False) + 1e-6

        # Copy computed stats to the registered buffers
        self.latent_mean.copy_(all_means.to(self.device))
        self.latent_std.copy_(all_stds.to(self.device))
        

    def infer(self, labels, beta=0.4, use_learned_stats=True):
        self.eval()
        batch_size = labels.size(0)

        with torch.no_grad():
            if use_learned_stats:
                # Get per-class mean and std for the requested labels
                labels_list = labels.cpu().tolist()

                # Use pre-computed latent stats for conditioning, basically better results
                mu_list = [self.latent_mean[l].cpu() for l in labels_list]
                std_list = [self.latent_std[l].cpu() for l in labels_list]
                
                mu = torch.stack(mu_list, dim=0).to(self.device) # (batch_size, 128)
                std = torch.stack(std_list, dim=0).to(self.device) # (batch_size, 128)
                
                # Stochastic sampling: z = mu + beta * eps * std
                eps = torch.randn_like(std)
                z = mu + beta * eps * std
            else:
                # sample from standard normal distribution N(0, 1)
                z = torch.randn(batch_size, 128, device=self.device)

            # encode the requested class labels
            labels = labels.to(self.device)
            y = self._one_hot_encode_labels(labels)

            z_conditioned = torch.cat((z, y), dim=1)

            output = self.decoder(z_conditioned)

            # Reshape to image format
            output = output.view(batch_size, 1, 28, 28)

            # Denormalize
            output = output * 0.5 + 0.5

            # Clip to valid pixel range [0, 1]
            output = output.clamp(0, 1)

        return output

    def _one_hot_encode_labels(self, labels):
        # Labels is a tensor of class labels
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, 10, device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        return one_hot

