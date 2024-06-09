import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from model import CBOW_Model
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from trainer import Trainer
import os
from dataloader import get_dataloader_and_vocab

torchtext.disable_torchtext_deprecation_warning()

def get_english_tokenizer():
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def save_vocab(vocab, model_dir: str):
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)

word_string = open('final_file.txt',encoding="utf8").read()

tokenizer = get_english_tokenizer()
word_list = tokenizer(word_string)

batch = []
idx = 0
while idx < len(word_list):
    sentence = " ".join(word_list[idx:idx+256])
    batch.append(sentence)
    idx += 256
    


def train(batch_size = 256,shuffle = True):
    train_dataloader, vocab = get_dataloader_and_vocab(
        word_list = word_list,
        batch = batch,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        word_list=word_list,
        batch = batch,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
    vocab_size = len(vocab.get_stoi())
    
    model = CBOW_Model(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.025)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(
        model=model,
        epochs=int(1e2),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_steps=500,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=1000,
        device=device,
        model_dir='checkpoints/',
    )
    
    trainer.train()
    
    trainer.save_model()
    save_vocab(vocab, 'checkpoints/')
    
if __name__ == '__main__':
    train()