# models.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from utils import *
from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_index):
        # raise Exception("Implement me")
        super().__init__()
        self.vocab_index = vocab_index
        self.vocab_size=len(self.vocab_index) 
        self.embedding = nn.Embedding(self.vocab_size, 256)
        self.positional_encoding = PositionalEncoding(d_model=256, num_positions=19, batched=False)    
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = nn.Linear(256, self.vocab_size)

    def forward(self, input_seq):
        input_seq = input_seq.transpose(0, 1)
        embedded = self.embedding(input_seq)
        encoded = self.positional_encoding(embedded)
        size = input_seq.size(0)
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        transformer_out = self.transformer(encoded, mask = mask)
        output = self.output_layer(transformer_out)
        return output

    def get_next_char_log_probs(self, context):
        # raise Exception("Implement me")
        if len(context) == 0:
            context = ' '
        
        self.eval()
        with torch.no_grad():
            trun_indices = context[-19:]
            input_indices = torch.LongTensor([[self.vocab_index.index_of(c) for c in trun_indices]])
            logits = self.forward(input_indices)
            next_char_logits = logits[0, -1]
            log_probs = nn.functional.log_softmax(next_char_logits, dim=-1)
        return log_probs.numpy()

    def get_log_prob_sequence(self, next_chars, context):
        # raise Exception("Implement me")
        if len(next_chars) == 0:
            return 0.0
        
        self.eval()
        log_prob = 0.0
        with torch.no_grad():
            for c in next_chars:
                log_probs = self.get_next_char_log_probs(context)
                idx = self.vocab_index.index_of(c)
                log_prob += log_probs[idx]
                context += c
        return log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # raise Exception("Implement me")
    model = NeuralLanguageModel(vocab_index=vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_data = [vocab_index.index_of(char) for char in train_text]
    # space_index = vocab_index.index_of(' ')

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        i = 0
        chunk_size = 20
        while i < (len(train_data)-chunk_size):
            chunk_start = max(0, i)
            chunk_end = chunk_start + chunk_size
            if train_text[chunk_start] != ' ':
                while chunk_start > 0 and train_text[chunk_start] != ' ':
                    chunk_start -= 1
                chunk_end = chunk_start + chunk_size

            input_seq = train_data[chunk_start+1:chunk_end]
            target_seq = train_data[chunk_start+2:chunk_end+1]
            optimizer.zero_grad()
            input_tensor = torch.LongTensor([input_seq])
            target_tensor = torch.LongTensor([target_seq])
            output = model(input_tensor)
            output = output.reshape(-1, 27)
            target_tensor = target_tensor.view(-1)

            loss = loss_fn(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            i += chunk_size
        avg_loss = total_loss / ((len(train_data) - chunk_size) // chunk_size)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch + 1}, Loss = {avg_loss}, Perplexity = {perplexity}")

    return model

