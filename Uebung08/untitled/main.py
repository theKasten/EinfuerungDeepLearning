from classify_spam import SpammClassifyer, SpamClassifierDataset
from torch.utils.data import DataLoader
import datasets

n_vocab = 2000
dataset = SpamClassifierDataset(datasets.load_dataset('sms_spam'), vocab_size=n_vocab)
dataloader = DataLoader(dataset=dataset,
                        batch_size=2**8,
                        shuffle=True,
                        num_workers=0)

EMBED_SIZE = 200
LR = 3e-4
EPOCHS = 10

# for input_word, is_spam_lable in dataloader:
#     print("----------------------")
#     print("Input Word:")
#     print(input_word)
#     print("Lable:")
#     print(is_spam_lable)
#     print("----------------------")

model = SpammClassifyer(n_vocab, EMBED_SIZE)
model.train(dataloader, EPOCHS, LR)
#model.plot_running_loss()