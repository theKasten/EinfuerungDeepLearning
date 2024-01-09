import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import torch
from torch import nn
from torch.utils.data import Dataset
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.decomposition import PCA

class SpammClassifyer(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.running_accuracy = []
        self.running_loss = []
        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_size),
            nn.Linear(embedding_size, round(embedding_size/2)),
            nn.BatchNorm1d(round(embedding_size/2)),
            nn.SiLU(True),
            nn.Linear(round(embedding_size/2), round(embedding_size/4)),
            nn.BatchNorm1d(round(embedding_size/4)),
            nn.SiLU(True),
            nn.Linear(round(embedding_size/4), 1),
            nn.BatchNorm1d(1),
            nn.SiLU(True)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        try:
            import torch_directml
            self.device = torch_directml.device()
        except:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            torch.set_num_threads(24) # 12
        self.to(self.device)

    def train(self, dataloader,  epochs, learning_rate):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.to(self.device)
        progress_bar = tqdm(range(epochs * len(dataloader)))
        self.model.train()
        self.running_loss = []
        for epoch in range(epochs):
            epoch_loss = 0
            for input_word, is_spam_lable in dataloader:
                input_word, is_spam_lable = input_word.to(self.device), is_spam_lable.to(self.device)
                optimizer.zero_grad()
                predicted_lable = self.forward(is_spam_lable)
                print(predicted_lable)
                loss = self.loss_fn(predicted_lable, input_word)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
            epoch_loss /= len(dataloader)
            self.running_loss.append(epoch_loss)
            #self.running_accuracy.append(self.accuracy(dataloader))

    def forward(self, input):
        return self.model(input)


class SpamClassifierDataset(Dataset):
    """
    Takes a HuggingFace dataset as an input, to be used for a Word2Vec dataloader.
    """
    def __init__(self, dataset, vocab_size, wsize=3, stemmer= SnowballStemmer('english'), sw = stopwords.words('english')):
        nltk.download('stopwords')

        print("Initialising Spam Dataloader")
        print(dataset)

        self.dataset = dataset
        self.vocab_size = vocab_size
        self.stemmer = stemmer
        self.sw = sw
        self.wsize = wsize
        self.dataset = self.dataset.map(self.split_tokens)
        self.counts = Counter([i for s in self.dataset['train']['all_tokens'] for i in s])
        self.counts = dict(self.counts.most_common(vocab_size))
        self.vocab = list(self.counts.keys())
        self.id2tok = dict(enumerate(self.vocab))
        self.tok2id = {token: id for id, token in self.id2tok.items()}
        self.dataset = self.dataset.map(self.remove_rare_tokens)
        self.dataset = self.dataset.map(self.windowizer)
        self.dataset = self.dataset['train']
        self.data = [i for s in self.dataset['moving_window'] for i in s]
        print("Done initialising Dataloader...")

    def split_tokens(self, row):
        row['all_tokens'] = [self.stemmer.stem(i) for i in
                             re.split(r" +",
                                      re.sub(r"[^a-z ]", "", # Entferne alle Zeichen a mit Ausnahme von Buchstaben und Leerzeichen!
                                             row['sms'].lower()))
                             if (i not in self.sw) and len(i)]
        return row

    def remove_rare_tokens(self, row):
        row['tokens'] = [t for t in row['all_tokens'] if t in self.vocab]
        return row

    def windowizer(self, row):
        """
        Windowizer function for Word2Vec. Converts sentence to sliding-window
        pairs.
        """
        doc = row['tokens']
        out = []
        for i, wd in enumerate(doc):
            target = self.tok2id[wd]
            window = [i+j for j in
                      range(-self.wsize, self.wsize+1, 1)
                      if (i+j>=0) &
                      (i+j<len(doc)) &
                      (j!=0)]

            out+=[(target, self.tok2id[doc[w]]) for w in window]
        row['moving_window'] = out
        return row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]