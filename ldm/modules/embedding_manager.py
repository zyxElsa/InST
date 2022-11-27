import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
import numpy as np
from ldm.modules.attention import CrossAttention
import PIL
from PIL import Image

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    #assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_per_img=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}

        self.init = True

        self.cond_stage_model = embedder

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        self.attention = Attentions(dim=token_dim, n_heads=8, d_head=64, dropout = 0.05) 

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)
    
        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)
                
            self.string_to_token_dict[placeholder_string] = token

    def forward(
            self,
            tokenized_text,
            embedded_text,            
            image_embeds,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            
            placeholder_embedding = self.attention(image_embeds.view(b,1,768).to(device), image_embeds.view(b,1,768).to(device)).view(1,768)   

            if self.max_vectors_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding.float()
            else: # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save({
                    "string_to_token": self.string_to_token_dict,
                    "attention": self.attention,
                    }, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print('find keys:',ckpt.keys())

        self.string_to_token_dict = ckpt["string_to_token"]

        if 'attention' in ckpt.keys():
            self.attention = ckpt["attention"]
        else:
            self.attention = None

    def embedding_parameters(self):
        return self.attention.parameters()

    def embedding_to_coarse_loss(self):        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]            
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss

class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x)
        x_2 = self.attn2(x_1, x)
        x_3 = self.net(x_2)
        return x_3
