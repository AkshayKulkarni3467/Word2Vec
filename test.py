
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from constants import MIN_WORD_FREQUENCY
from constants import MAX_SEQUENCE_LENGTH
from constants import CBOW_N_WORDS
import torch
from torch.utils.data import DataLoader
from functools import partial


def get_english_tokenizer():
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

tokenizer = get_english_tokenizer()


def build_vocab(data_iter, tokenizer):
        
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

vocab = torch.load('checkpoints/vocab.pt')

value = torch.Tensor([[vocab.lookup_indices(['shredded'])[0]
,vocab.lookup_indices(['the'])[0]
,vocab.lookup_indices(['letters'])[0]
,vocab.lookup_indices(['in'])[0]
,vocab.lookup_indices(['food'])[0]
,vocab.lookup_indices(['shredded'])[0]
,vocab.lookup_indices(['processor'])[0]
,vocab.lookup_indices(['who'])[0]]]).to('cpu')
value = value.to(torch.int64)


model = torch.load('checkpoints/model.pt')
model.to('cpu')

return1 = model(value)
return1 = return1.detach().numpy()[0]
print(f'The received Embedding for context window 4 is : {return1}')
