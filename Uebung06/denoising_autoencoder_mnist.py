#! /usr/bin/env python3
"""Spaghetti code to be turned into a beautiful lasagna."""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

IMAGESIZE = 28

class autoencoder_mnist:
    # Data Loaders
    usps_data_loader = None
    usps_test_loader = None
    mnist_data_loader = None
    mnist_test_loader = None
    fashion_data_loader = None
    fashion_test_loader = None

    # Model
    model = None
    device = None
    reconstruction_criterion = None
    classification_criterion = None
    optimizer = None

    # Visualisation
    loss_values = []
    accuracy_values_test = []
    accuracy_values_train = []

    def run(self):
        self.check_for_gpu()
        self.setup_datasets_and_loaders()

        # Training:
        self.instantiate_model()
        self.start_training()

        #self.safe_results_to_disk()

        self.start_visualization_of_results()

    def start_epoch(self, data_loader, rho, epoch, num_epochs):
        total_loss = 0
        #noise_factor = 0.3 #0.6 - 0.6*(1 - epoch/num_epochs)
        for img, labels in data_loader:
            img = img.to(self.device)
            labels = labels.to(self.device)
            # forward pass
            #noisy_img = add_noise(img, noise_factor).to(device)
            latent, reconstructed, label_pred = self.model(img)
            #latent, reconstructed, label_pred = model(noisy_img)
            # compute losses
            rho_hat = torch.mean(latent, dim=0)
            sparsity_penalty = self.kl_divergence(rho, rho_hat).sum()
            reconstruction_loss = self.reconstruction_criterion(reconstructed, img)
            classification_loss = self.classification_criterion(label_pred, labels)

            #test_accuracy()
            #print("Epoch %d classification_loss %d", epoch, classification_loss)
            #print("Epoch %d Accuracy on Test %d", epoch, classification_loss)

            loss = reconstruction_loss + classification_loss + 0.01*sparsity_penalty
            total_loss += loss.item()
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Average loss for this epoch
        avg_loss = total_loss / len(data_loader)
        self.loss_values.append(avg_loss)
        # Calc and log Accuracy on Test and Train data
        self.accuracy_test = self.accuracy(self.mnist_test_loader)
        self.accuracy_train = self.accuracy(self.mnist_data_loader)
        self.accuracy_values_test.append(self.accuracy_test)
        self.accuracy_values_train.append(self.accuracy_train)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy Test: {self.accuracy_test:.2f}%, Accuracy Train: {self.accuracy_train:.2f}%')

    def display_results(self, test_images, reconstructed, true_label, label_pred):
            # Display results
            fig, axs = plt.subplots(2, 10, figsize=(10, 2))
            for i in range(10):
                axs[0, i].imshow(np.reshape(test_images[i], (IMAGESIZE, IMAGESIZE)), cmap='gray_r')
                axs[1, i].imshow(np.reshape(reconstructed[i], (IMAGESIZE, IMAGESIZE)), cmap='gray_r')
                axs[0, i].set_xlabel(true_label[i])
                axs[1, i].set_xlabel(label_pred[i])
                axs[0, i].set_xticks([])
                axs[0, i].set_yticks([])
                axs[1, i].set_xticks([])
                axs[1, i].set_yticks([])

            fig.suptitle('Top: Noisy Images, Bottom: Reconstructed Images')
            plt.tight_layout()
            plt.savefig('denoising-example.png')
            plt.show()
            plt.clf()

            # Plotting the training loss
            plt.figure(figsize=(10, 4))
            plt.plot(self.loss_values, label='Training Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('dlae-loss-curve.png')
            plt.clf()

            # Plotting the Accuracy
            plt.figure(figsize=(10, 4))
            plt.title('Accuracy Over Epochs')
            #
            plt.plot(self.accuracy_values_test, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            #
            plt.plot(self.accuracy_values_train, label='Training Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            #
            plt.legend()
            plt.grid(True)
            plt.savefig('dlae-accuracy-curve.png')
            plt.clf()


            ### ==== Visualization of the latent space
            self.model.eval()

            # Collect all the latent space representations
            latent_space = []
            labels = []
            with torch.no_grad():
                for data, label in self.mnist_data_loader:#TODO: siehe oben, geht das so?
                    data = data.to(self.device)
                    latent = self.model.encoder(data).cpu().numpy()
                    latent_space.append(latent)
                    labels.append(label.numpy())

            latent_space = np.concatenate(latent_space, axis=0)
            labels = np.concatenate(labels, axis=0)

            # reduce dimensions linearly
            pca = PCA(n_components=3)
            latent_space_reduced = pca.fit_transform(latent_space)

            # plot it like it's hot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(latent_space_reduced[:, 0],
                                 latent_space_reduced[:, 1],
                                 latent_space_reduced[:, 2],
                                 c=labels, cmap='viridis', alpha=0.2)
            ax.set_title("3D Visualization of Latent Space with PCA")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Labels')
            plt.savefig("dlae-latent-space.png")
            plt.clf()

    def configure_cpu_parallel(self):
        # Configure CPU parallelism
        torch.set_num_threads(12)

    def transform_view(self):
        return lambda x: x.view(-1)

    def start_training_loop(self, device, mnist_data_loader, model, optimizer, mnist_test_loader, reconstruction_criterion, classification_criterion):
        # Training Loop
        num_epochs = 5
        rho = 0.1 # mean activation of latent neurons that we want
        data_loader = mnist_data_loader
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            #noise_factor = 0.3 #0.6 - 0.6*(1 - epoch/num_epochs)
            for img, labels in data_loader:
                img = img.to(device)
                labels = labels.to(device)
                # forward pass
                #noisy_img = add_noise(img, noise_factor).to(device)
                latent, reconstructed, label_pred = model(img)
                #latent, reconstructed, label_pred = model(noisy_img)
                # compute losses
                rho_hat = torch.mean(latent, dim=0)
                sparsity_penalty = self.kl_divergence(rho, rho_hat).sum()
                reconstruction_loss = reconstruction_criterion(reconstructed, img)
                classification_loss = classification_criterion(label_pred, labels)

                #test_accuracy()
                #print("Epoch %d classification_loss %d", epoch, classification_loss)
                #print("Epoch %d Accuracy on Test %d", epoch, classification_loss)

                loss = reconstruction_loss + classification_loss + 0.01*sparsity_penalty
                total_loss += loss.item()
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Average loss for this epoch
            avg_loss = total_loss / len(data_loader)
            self.loss_values.append(avg_loss)
            # Calc and log Accuracy on Test and Train data
            accuracy_test = self.accuracy(mnist_test_loader)
            accuracy_train = self.accuracy(mnist_data_loader)
            self.accuracy_values_test.append(accuracy_test)
            self.accuracy_values_train.append(accuracy_train)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy Test: {accuracy_test:.2f}%, Accuracy Train: {accuracy_train:.2f}%')

    # Custom KL Divergence, to be used for sparsity penalty
    def kl_divergence(self, rho, rho_hat):
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    def accuracy(self, loader):
        global labels, _
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, _, predicted_labels = self.model(images)
                _, predicted = torch.max(predicted_labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
        #print(f'Accuracy of the model on the {total} test images: {100 * correct / total}%')

    def start_visualization_of_results(self):
        ### ==== Visualization of results

        # Look at some images
        self.model.eval()
        test_images = None
        reconstructed = None
        labels = None
        with torch.no_grad():
            for img, labels in self.mnist_data_loader:#TODO: Ist das so in ordnung? Vorher wurde data_loader = mnist_dataloader in trian loop gesetzt
                img = img.to(self.device)
                #noisy_img = add_noise(img, 0.2).to(device)
                #_, reconstructed, label_pred = model(noisy_img)
                _, reconstructed, label_pred = self.model(img)
                break # Just show one batch of images

        # Convert to numpy for visualization
        #test_images = noisy_img.cpu().detach().numpy()
        test_images = img.cpu().detach().numpy()
        reconstructed = reconstructed.cpu().detach().numpy()
        label_pred = label_pred.cpu().detach().numpy()
        label_pred = np.argmax(label_pred, axis=1)
        true_label = labels.cpu().detach().numpy()
        self.display_results(test_images, reconstructed, true_label, label_pred)

    def check_for_gpu(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Working on {self.device}.')
        self.configure_cpu_parallel()

    def setup_datasets_and_loaders(self):
        # Dataset setup
        transform_usps = transforms.Compose([
            transforms.Resize((IMAGESIZE, IMAGESIZE)),  # Upscaling from 16x16
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(self.transform_view())
        ])
        transform_mnist = transforms.Compose([ # already 28x28
            transforms.ToTensor(),
            transforms.Lambda(self.transform_view())
        ])

        # Dataloaders setup
        usps_data = datasets.USPS(root='D:/Datasets', train=True, download=False, transform=transform_usps)#Manuell gedownloaded: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps
        usps_test = datasets.USPS(root='D:/Datasets', train=False, download=False, transform=transform_usps)#Manuell gedownloaded: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps

        mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        fashion_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
        fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_mnist)
        self.usps_data_loader = DataLoader(dataset=usps_data, batch_size=512,
                                           num_workers=0, shuffle=True)
        self.usps_test_loader = DataLoader(dataset=usps_test, batch_size=512,
                                           num_workers=0, shuffle=False)
        self.mnist_data_loader = DataLoader(dataset=mnist_data, batch_size=1024,
                                            num_workers=0, shuffle=True)
        self.mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=1024,
                                            num_workers=0, shuffle=False)
        self.fashion_data_loader = DataLoader(dataset=fashion_data, batch_size=2048,
                                              num_workers=0, shuffle=True)
        self.fashion_test_loader = DataLoader(dataset=fashion_test, batch_size=2048,
                                              num_workers=0, shuffle=False)

    def instantiate_model(self):
        # Instantiate the model
        self.model = DenoisingLabelAutoencoder(latent_dim=12).to(self.device)
        #model = torch.load('dlae.pth')
        # Loss function and Optimizer
        self.reconstruction_criterion = nn.MSELoss() # mean squared error
        self.classification_criterion = nn.NLLLoss() # neg. log likelihood
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def start_training(self, num_epochs=5, rho=0.1): # rho mean activation of latent neurons that we want
        # Training Loop
        self.loss_values = []
        accuracy_values_train = []
        accuracy_values_test = []
        data_loader = self.mnist_data_loader
        self.model.train()
        for epoch in range(num_epochs):
            self.start_epoch(data_loader, rho, epoch, num_epochs)

    def safe_results_to_disk(self):
        #save weights to disk:
        torch.save(self.model, 'dlae.pth')

        #save ONNX to disk:
        self.model.eval()
        dummy_input = torch.randn(1, 1, 28**2, device="cuda")
        torch.onnx.export(self.model, dummy_input, "dlae.onnx",  export_params=True)


### === Denoising Autoencoder Definition ===
class DenoisingLabelAutoencoder(nn.Module):
    """Implementation of a denoising label autoencoder with some experiments"""
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IMAGESIZE**2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.SiLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.SiLU(True),
            nn.Linear(16, latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.SiLU(True),
            nn.Linear(16, 32),
            nn.SiLU(True),
            nn.Linear(32, 64),
            nn.SiLU(True),
            nn.Linear(64, 128),
            nn.SiLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(True),
            nn.Linear(256, IMAGESIZE**2),
            nn.Sigmoid()
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, data):
        latent = self.encoder(data)
        reconstruction = self.decoder(latent)
        label_prediction = self.label_predictor(latent)
        return latent, reconstruction, label_prediction

