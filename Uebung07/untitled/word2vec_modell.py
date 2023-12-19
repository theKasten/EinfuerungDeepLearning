import string

import torch as torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Naive Test Daten
tokenize_me = "Hier steht ein Satz, an dem man testen kann, ob der Tokenizer tut was er tun soll. Und zur Sicherheit gleich noch ein zweiter."

# Hyperparameter
N_INPUT_DIM = 2000
LATENT_DIM = 200
withBias = False

class Word2vec_modell_impl:

    # Model
    model = None
    device = None
    #reconstruction_criterion = None
    #classification_criterion = None
    #optimizer = None

    def run(self, epochs, rho):
        self.check_for_gpu()
        #self.setup_datasets_and_loaders(True)# SSL Error beim downloaden der usps Datasets

        self.instantiate_model()
        #self.start_training(epochs, rho)

        #self.safe_results_to_disk()

    def check_for_gpu(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Working on {self.device}.')
        self.configure_cpu_parallel()

    def configure_cpu_parallel(self):
        # Configure CPU parallelism
        torch.set_num_threads(24)#12

    def instantiate_model(self):
        # Instantiate the model
        self.model = Word2vec_autoencoder(latent_dim=12).to(self.device)
        #model = torch.load('dlae.pth')
        # Loss function and Optimizer
        self.reconstruction_criterion = nn.MSELoss() # mean squared error
        self.classification_criterion = nn.NLLLoss() # neg. log likelihood
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def one_hot_encode(input, size):
        vec = torch.zeros(size).float()
        vec[input] = 1.0
        return vec

    def create_tokens(self, input_text, max_tokens):
        input_text = self.all_capital(input_text)
        input_text = self.remove_punctuation_marks(input_text)
        print("Creating Tokens from: " + input_text)
        token_index = 0
        token_list = [""]*max_tokens
        for char_index in range(len(input_text)):
            if(input_text[char_index] == ' '):
                token_candidate = token_list[token_index]
                print("New Token candidate: " + token_candidate)
                if(self.last_token_has_been_in_list(token_list, token_candidate)):
                    print("Token already existed")
                    token_list[token_index] = ""
                else:
                    print("New Token added: " + token_candidate)
                    token_index += 1

                if(token_index >= max_tokens):
                    return token_list
            else:
                token_list[token_index] += input_text[char_index]
        # Maximale Tokenzahl wurde mit Satzlänge nicht erreicht, return
        return token_list

    def last_token_has_been_in_list(self, token_list, new_token):
        found = token_list.count(new_token)
        if found == 1:
            return False
        else:
            print("Token found %d times\n", found)
            return True
        # if len(token_list) == 1:
        #     return True
        # new_token = token_list[len(token_list) - 1]
        # print("New Token: " + str(new_token) + "\n")
        # for index in range(0, len(token_list) - 2):
        #     if token_list[index] == new_token:
        #         return True
        # # Das Letzte Token der Liste kam nicht erneut in der Liste vor
        # return False;

    def all_capital(self, input_text):
        input_text_big = ""
        for char in input_text:
            input_text_big += char.upper()
        return input_text_big

    def remove_punctuation_marks(self, input_text):
        translator = str.maketrans("", "", string.punctuation)
        return input_text.translate(translator)


### === Denoising Autoencoder Definition ===
class Word2vec_autoencoder(nn.Module):
    """Implementation of a denoising label autoencoder with some experiments"""
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_INPUT_DIM, round(N_INPUT_DIM/2), bias=self.withBias),
            nn.BatchNorm1d(round(N_INPUT_DIM/2)),
            nn.NLLLoss(True),#Negative Log likelihood loss

            nn.Linear(round(N_INPUT_DIM/2), LATENT_DIM, bias=self.withBias),
            nn.BatchNorm1d(LATENT_DIM),
            nn.NLLLoss(True),#Negative Log likelihood loss

            # nn.Linear(IMAGESIZE**2, 256),
            # nn.BatchNorm1d(256),
            # nn.SiLU(True),
            #
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.SiLU(True),
            #
            # nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.SiLU(True),
            #
            # nn.Dropout(0.2),
            # nn.Linear(64, 32),
            # nn.SiLU(True),
            #
            # nn.Dropout(0.1),
            # nn.Linear(32, 16),
            # nn.SiLU(True),
            #
            # nn.Linear(16, latent_dim),
            # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, round(N_INPUT_DIM/2), bias=self.withBias),
            nn.NLLLoss(True),

            nn.Linear(round(N_INPUT_DIM/2),bias=self.withBias),
            nn.NLLLoss()
            #
            # nn.Linear(latent_dim, 16),
            # nn.SiLU(True),
            #
            # nn.Linear(16, 32),
            # nn.SiLU(True),
            #
            # nn.Linear(32, 64),
            # nn.SiLU(True),
            #
            # nn.Linear(64, 128),
            # nn.SiLU(True),
            #
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            # nn.SiLU(True),
            #
            # nn.Linear(256, IMAGESIZE**2),
            # nn.Sigmoid()
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


print("Starting tokeniser Test...\n")
model = Word2vec_modell_impl()
tokens = model.create_tokens(tokenize_me, 25)
for token in tokens:
    print("[" + str(token) + "]\n")