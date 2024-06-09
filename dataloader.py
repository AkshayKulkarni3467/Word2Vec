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


def build_vocab(data_iter, tokenizer):
        
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab



def collate_cbow(batch, text_pipeline):

    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


    
def get_dataloader_and_vocab(word_list,batch, batch_size, shuffle):

    tokenizer = get_english_tokenizer()


    vocab = build_vocab(word_list, tokenizer)
        
    text_pipeline = lambda x: vocab(tokenizer(x))

    
    collate_fn = collate_cbow
    

    dataloader = DataLoader(
        batch,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab

