import torch.nn as nn
import torch


class ConditionalAutoEncoder(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()

        # You can change the network structure
        # You may add or remove any elements
        #   other than the save, load, forward,
        #   and infer functions itself

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 + 10, 512), # Added 10 for one-hot encoded class labels
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128 + 10, 256), # Added 10 for one-hot encoded class labels
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )
        
        # Register buffers for per-class latent statistics (saved/loaded with model state)
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
        model = ConditionalAutoEncoder(device=device)
        model.load_state_dict(model_data["state_dict"])
        return model

    def forward(self, images, classes):
        # This forward operation only learns how to encode and decode an input digit image
        # You should apply some changes to enable model to learn how to encode and decode images regarding their classes

        # Concatenate one-hot encoded class labels to the input images
        input_tensor = images.view(-1, 28 * 28)
        y = self._one_hot_encode_labels(classes)
        input_tensor = torch.cat((input_tensor, y), dim=1)

        z = self.encoder(input_tensor)

        # Concatenate one-hot encoded class labels to the latent vector
        z = torch.cat((z, y), dim=1)

        return self.decoder(z).reshape(images.shape)

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
                
                # Collect latent vectors for this class
                latent_vectors = []
                for idx in selected_indices:
                    image, label = train_dataset[idx]
                    image = image.unsqueeze(0).to(self.device)
                    label_tensor = torch.tensor([label], device=self.device)
                    
                    # Encode to get latent vector
                    input_tensor = image.view(-1, 28 * 28)
                    y = self._one_hot_encode_labels(label_tensor)
                    input_tensor = torch.cat((input_tensor, y), dim=1)
                    z = self.encoder(input_tensor)
                    latent_vectors.append(z.detach().cpu())
                
                # Stack and compute statistics on CPU
                latent_vectors = torch.cat(latent_vectors, dim=0)
                all_means[class_idx] = latent_vectors.mean(dim=0)
                all_stds[class_idx] = latent_vectors.std(dim=0, unbiased=False) + 1e-6
                
                print(f"  Class {class_idx}: {num_samples} samples, mean range [{all_means[class_idx].min():.3f}, {all_means[class_idx].max():.3f}]")
        
        # Copy computed stats to the registered buffers
        self.latent_mean.copy_(all_means.to(self.device))
        self.latent_std.copy_(all_stds.to(self.device))
        
        print(f"Computed latent statistics from {sample_percentage*100:.1f}% samples per class")

    def infer(self, labels, beta=0.4):
        self.eval()
        batch_size = labels.size(0)
        
        with torch.no_grad():
            # Get per-class mean and std for the requested labels
            # Do all operations on CPU to avoid MPS indexing issues
            labels_list = labels.cpu().tolist()
            
            mu_list = [self.latent_mean[l].cpu() for l in labels_list]
            std_list = [self.latent_std[l].cpu() for l in labels_list]
            
            mu = torch.stack(mu_list, dim=0).to(self.device)  # (batch_size, 128)
            std = torch.stack(std_list, dim=0).to(self.device)  # (batch_size, 128)
            labels = labels.to(self.device)
            
            # Stochastic sampling: z = mu + beta * eps * std
            eps = torch.randn_like(std)
            z = mu + beta * eps * std
            
            # Concatenate one-hot encoded class labels to the latent vector
            y = self._one_hot_encode_labels(labels)
            z_conditioned = torch.cat((z, y), dim=1)
            
            # Decode
            output = self.decoder(z_conditioned)
            
            # Reshape to image format
            output = output.view(batch_size, 1, 28, 28)
            
            # Clip to valid pixel range
            output = output.clamp(0, 1)
        
        return output

    def _one_hot_encode_labels(self, labels):
        # Labels is a tensor of class labels
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, 10, device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        return one_hot
