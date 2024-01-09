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

class Word2VecDataset(Dataset):
    """
    Takes a HuggingFace dataset as an input, to be used for a Word2Vec dataloader.
    """
    def __init__(self, dataset, vocab_size, wsize=3, stemmer= SnowballStemmer('english'), sw = stopwords.words('english')):
        nltk.download('stopwords')
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
    


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.running_accuracy = []
        self.running_loss = []
        self.model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_size),
            nn.Linear(embedding_size, vocab_size, bias=False)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        try:
            import torch_directml
            self.device = torch_directml.device()
        except:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            torch.set_num_threads(24) # 12
        self.to(self.device)

    def forward(self, input):
        return self.model(input)
    
    def get_wordvecs(self):
        return self.model[1].weight.cpu().detach().numpy()
    def set_wordvecs(self, wordvecs):
        self.model[1].weight = nn.Parameter(torch.tensor(wordvecs).float().to(self.device))

    def get_distance_matrix(self, metric='cosine'):
        wordvecs = self.get_wordvecs()
        dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
        return dist_matrix

    def get_k_similar_words(self, word, dataset, k=10):
        dist_matrix = self.get_distance_matrix('cosine')
        idx = dataset.tok2id[word]
        dists = dist_matrix[idx]
        ind = np.argpartition(dists, k)[:k+1]
        ind = ind[np.argsort(dists[ind])][1:]
        out = [(i, dataset.id2tok[i], dists[i]) for i in ind]
        return out

    def train(self, dataloader,  epochs, learning_rate):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.to(self.device)
        progress_bar = tqdm(range(epochs * len(dataloader)))
        self.model.train()
        self.running_loss = []
        for epoch in range(epochs):
            epoch_loss = 0
            for center, context in dataloader:
                center, context = center.to(self.device), context.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(context)
                loss = self.loss_fn(logits, center)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
            epoch_loss /= len(dataloader)
            self.running_loss.append(epoch_loss)
            #self.running_accuracy.append(self.accuracy(dataloader))

    def accuracy(self, loader):
        global labels, _
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # for tokens in loader:
            #     tokens = tokens.to(self.device)
            #     predictions = self.model(tokens)
            #     predicted = torch.max(predictions.data, 1)
            #     total +=

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, _, predicted_labels = self.model(images)
                _, predicted = torch.max(predicted_labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def plot_running_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.running_loss)
        plt.show()
        # fig1, ax1 = plt.subplots()
        # ax1.plot(self.running_accuracy)
        # plt.show()
        print("Last Loss Value: " + str(self.running_loss[-1]) + "\n")

    def visualize_latent_vectors(self, annotate=False, dataset=None, tokens=None):
        wordvecs = self.get_wordvecs()
        pca = PCA(n_components=2)
        pca.fit(wordvecs)
        wordvecs_pca = pca.transform(wordvecs)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.scatter(wordvecs_pca[:,0], wordvecs_pca[:,1], alpha=0.7, marker = '.')
        if annotate:
            if (dataset != None and tokens != None):
                dmat = self.get_distance_matrix('cosine')
                for token in tokens:
                    if token in dataset.vocab:
                        ax.annotate(token, (wordvecs_pca[dataset.tok2id[token],0], wordvecs_pca[dataset.tok2id[token],1]))
                        ax.scatter(wordvecs_pca[dataset.tok2id[token],0], wordvecs_pca[dataset.tok2id[token],1], c='orange')
                        similar_words = [t[1] for t in self.get_k_similar_words(token, k=3, dataset=dataset)]
                        for similar_word in similar_words:
                            ax.annotate(similar_word, (wordvecs_pca[dataset.tok2id[similar_word],0], wordvecs_pca[dataset.tok2id[similar_word],1]) )
                            ax.scatter(wordvecs_pca[dataset.tok2id[similar_word],0], wordvecs_pca[dataset.tok2id[similar_word],1], marker='o', c='yellow') 
        plt.show()   


