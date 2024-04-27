import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if self.config.use_bert:
            path = os.path.join(os.getcwd(), "data/train/word2int.tsv")
            word2int_df = pd.read_table(path)
            # Convert DataFrame to dictionary for faster lookup
            self.int2word = word2int_df.set_index('int')['word'].to_dict()
            self.BERTtokenizer = AutoTokenizer.from_pretrained(config.bert_model)
            self.word_embedding = AutoModel.from_pretrained(config.bert_model).to(device)
            # for param in self.word_embedding.parameters():
            #     param.requires_grad = False
        elif pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        if self.config.use_bert:
            titles = self.recover_words(news["title"])
            with torch.no_grad():
                try:
                    tokenized_titles = self.BERTtokenizer(titles, 
                                                        padding=True, 
                                                        return_tensors='pt',
                                                        is_split_into_words=True).to(device)
                    embedding = self.word_embedding(**tokenized_titles).last_hidden_state.to(device)
                except Exception as e:
                    print(f"An error occurred during tokenization: {e}")
                    print("Titles:", titles)
                    raise

        else:
            embedding = self.word_embedding(news["title"].to(device))

        news_vector = F.dropout(embedding,
                                p=self.config.dropout_probability,
                                training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
    

    def recover_words(self, title: torch.tensor):
        return [[str(self.int2word.get(int(idx), " ")) for idx in title_batch] for title_batch in title]