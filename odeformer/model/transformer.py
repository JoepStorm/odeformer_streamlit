# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import numpy as np
# import json

from odeformer.model.embedders import TwoHotEmbedder

N_MAX_POSITIONS = 4096  # maximum input sequence length


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None, use_two_hot=False):
    if use_two_hot:
        m = TwoHotEmbedder(num_embeddings, embedding_dim, padding_idx=padding_idx)
    else:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None, use_cache=False, return_attn=False):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return (

                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)

        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)

        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (
                (mask == 0).view(mask_reshape).expand_as(scores)
            )  # (bs, n_heads, qlen, klen)
            scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (bs, n_heads, qlen, klen)
        return_weights = weights#.detach().cpu()
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        if return_attn:
            # Compute value-scaled attention:
            v_norms = torch.norm(v, p=2, dim=-1, keepdim=True)  # (bs, n_heads, klen, 1)
            info_weighted_attn = weights * v_norms.transpose(2, 3)  # (bs, n_heads, qlen, klen)
            return self.out_lin(context), return_weights, info_weighted_attn
        return self.out_lin(context)


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout, activation):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.activation = getattr(F, activation)
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = self.activation(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerModel(nn.Module):

    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.use_cross_attention = params.use_cross_attention
        self.activation = params.activation
        if self.is_decoder:
            self.use_two_hot = params.use_two_hot
        self.apex = params.nvidia_apex

        # dictionary
        # for encoder: env.float_id2word, this only includes float_words
        # for decoder: 
        #    - if two-hot: 
        #            constant_id2word + equation_id2word, which includes equation_words but not float_words
        #    - else: 
        #            equation_id2word, which includes equation_words and float_words
        self.id2word = id2word 
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]

        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words

        # model parameters
        self.dim = (
            params.enc_emb_dim if is_encoder else params.dec_emb_dim
        )  # 512 by default
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = (
            params.n_enc_heads if is_encoder else params.n_dec_heads
        )  # 8 by default
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"

        # embeddings

        if positional_embeddings is None or positional_embeddings in ["alibi","none"]:
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(
                self.n_words, 
                self.dim, 
                padding_idx=self.pad_index, 
                #  only use two-hot in decoder and only if asked for
                use_two_hot=((not is_encoder) and params.use_two_hot),
            )
        else:
            self.embeddings = None
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                    activation = self.activation,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        self.cache = None

        # output layer
        if self.with_output:
            assert not self.use_prior_embeddings
            self.proj = nn.Linear(
                self.dim, self.n_words, bias=True
            )  ##added index for eos and tab
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight

        self.ignore_enc_layers = []
        self.k_to_store = 0

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = (
                torch.arange(src_len.max(), dtype=torch.long, device=lengths.device)
                < src_len[:, None]
            )

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # all layer outputs
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []
            self.intermediate_tokens = []
            self.topk = []
            self.stored_attn_weights = {"decoder": [], "encoder": [], "cross_attention": []}
            self.stored_scaled_attn_weights = {"decoder": [], "encoder": [], "cross_attention": []}

        # embeddings
        if not self.use_prior_embeddings:
            tensor = self.embeddings(x)
        else:
            tensor = x

        if not self.use_cross_attention and self.is_decoder and src_enc is not None:
            src_enc = src_enc.mean(dim=1)  # B,D
            tensor[:,0,:] = src_enc

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # # should we show token charts
        # with open("tmp_files/plot_token_charts.txt", "r") as file:
        #     plot_token_charts = str(file.read()).strip()
        #     plot_token_charts = True if plot_token_charts == "True" else False
        #     # print("plot_token_charts: ", plot_token_charts, type(plot_token_charts))

        # # should we store attentions
        # with open("tmp_files/store_attentions.txt", "r") as file:
        #     store_attentions = str(file.read()).strip()
        #     store_attentions = True if store_attentions == "True" else False
        #     # print("store_attentions: ", store_attentions, type(store_attentions))

        # with open("tmp_files/topk.txt", "r") as file:
        #     k = int(file.read())
        #
        # with open("tmp_files/ignore_enc_layers.txt", "r") as file:
        #     ignore_enc_layers = list(map(int, str(file.read()).split()))

        # transformer layers
        for i in range(self.n_layers):

            # ignore some encoder layers
            if not self.is_decoder and i in self.ignore_enc_layers:
                print(f"Skipping encoder layer {i}...")
                continue

            # self attention
            self.attentions[i].cache = self.cache
            attn, attn_weights, scaled_attn_weights = self.attentions[i](tensor, attn_mask, use_cache=use_cache, return_attn=True)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # store self-attention weights
            if self.is_decoder:
                self.stored_attn_weights["decoder"].append(attn_weights.detach().cpu()) #.numpy().tolist())
                self.stored_scaled_attn_weights["decoder"].append(scaled_attn_weights.detach().cpu()) #.numpy().tolist())
            else:
                self.stored_attn_weights["encoder"].append(attn_weights.detach().cpu()) #.numpy().tolist())
                self.stored_scaled_attn_weights["encoder"].append(scaled_attn_weights.detach().cpu()) #.numpy().tolist())

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None and self.use_cross_attention: 
                self.encoder_attn[i].cache = self.cache
                attn, cross_attn_weights, scaled_cross_attn_weights = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache, return_attn=True
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
                # store cross-attention weights
                self.stored_attn_weights["cross_attention"].append(cross_attn_weights.detach().cpu()) #.numpy().tolist())
                self.stored_scaled_attn_weights["cross_attention"].append(scaled_cross_attn_weights.detach().cpu()) #.numpy().tolist())

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())
                #print(i, tensor.detach().cpu())

            if self.STORE_OUTPUTS and self.is_decoder:
                #print(tensor.size()) # [x, 1, 512]
                reshaped_tensor = tensor.view(-1, self.dim) # [x, 512]
                lens_tokens, topk = self.decode_logits(reshaped_tensor, self.k_to_store)
                self.intermediate_tokens.append((i, lens_tokens))
                self.topk.append(topk)

        # if plot_token_charts:
        # self.plot_tokens()
        #     self.store_in_json()
        #     self.store_topk()
        # if store_attentions:
        #     self.store_attn_weights()
        #     self.store_scaled_attn_weights()

        # update cache length
        if use_cache:
            self.cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor
    
    def decode_logits(self, tensor, k=10):
        decoded_tokens = []
        topk = []
        for step_logits in tensor:
            lens_scores = self.proj(step_logits)
            lens_probs = F.softmax(lens_scores, dim=-1)
            token_idx = torch.argmax(lens_probs, dim=-1).item()
            token = self.id2word.get(token_idx, "<UNK>")  # fallback to <UNK>

            # get the top k tokens and their probabilities
            if k!=0:
                topk_probs, topk_idx = torch.topk(lens_probs, k)
                topk_tokens = [self.id2word.get(idx.item(), "<UNK>") for idx in topk_idx]
                topk.append((topk_tokens, topk_probs.tolist()))

            decoded_tokens.append(token)

        return decoded_tokens, topk
    
    # def plot_tokens(self):
    #     data = self.intermediate_tokens
    #     id2word_size = self.n_words
    #     plot_title = "Token Table Chart"
    #     # be_suspicious = True
    #     # with open("imp_params.txt", "r") as file:
    #         # be_suspicious = str(file.read()).strip()
    #         # be_suspicious = True if be_suspicious == "True" else False
    #         # print("be_suspicious: ", be_suspicious, type(be_suspicious))
    #
    #     # ensure we plot relevant decoder blocks only
    #     if len(data) == 0:
    #         print("Skipping encoder block...")
    #     elif len(data[0][1]) == 1:
    #         print("Skipping suspicious single beam business...")
    #
    #     else:
    #         row_headers = [f"decoder {i}" for i in range(len(data))]
    #         col_headers = [f"beam {i}" for i in range(len(data[0][1]))]
    #         rows = len(data[0][1])
    #         cols = len(data)
    #         norm = mcolors.Normalize(vmin=0, vmax=id2word_size - 1)
    #         cmap = plt.cm.get_cmap('tab20', id2word_size)
    #
    #         fig, ax = plt.subplots(figsize=(cols, rows))
    #         ax.axis('tight')
    #         ax.axis('off')
    #
    #         # preparing the table's data
    #         table_data = []
    #         table_colors = []
    #
    #         for i in range(cols):
    #             row = []
    #             color_row = []
    #             for j in range(rows):
    #                 token = data[i][1][j]
    #                 row.append(token)
    #                 color_row.append(cmap(norm(self.word2id.get(token, 0))))
    #             table_data.append(row)
    #             table_colors.append(color_row)
    #
    #         # making the table plot
    #         table = plt.table(
    #             cellText=table_data,
    #             cellColours=table_colors,
    #             rowLabels=row_headers,
    #             colLabels=col_headers,
    #             loc='center',
    #             cellLoc='center',
    #         )
    #         table.auto_set_font_size(False)
    #         table.set_fontsize(10)
    #         plt.title(plot_title)
    #         plt.show()

    # def store_in_json(self):
    #     new_value = self.intermediate_tokens
    #     if len(new_value) != 0 and len(new_value[0][1]) != 1:
    #         with open("tmp_files/all_intermediate_tokens.json", "r") as file:
    #             data = json.load(file)
    #         num_pairs = len(data)
    #         new_key = "token_"+str(num_pairs)
    #         data[new_key] = new_value
    #         with open("tmp_files/all_intermediate_tokens.json", "w") as file:
    #             json.dump(data, file, indent=4)

    # def store_attn_weights(self):
    #     new_value = self.stored_attn_weights
    #     with open("tmp_files/all_stored_attentions.json", "r") as file:
    #         data = json.load(file)
    #     num_pairs = len(data)
    #     if num_pairs == 0 and len(new_value["decoder"]) == 0 and len(new_value["cross_attention"]) == 0:
    #         new_key = "encoder"
    #         data [new_key] = new_value["encoder"]
    #     else:
    #         new_key = "token_"+str(num_pairs-1)
    #         _ = new_value.pop("encoder")
    #         data[new_key] = new_value
    #     with open("tmp_files/all_stored_attentions.json", "w") as file:
    #         json.dump(data, file, indent=4)
    #
    # def store_scaled_attn_weights(self):
    #     new_value = self.stored_scaled_attn_weights
    #     with open("tmp_files/all_stored_scaled_attentions.json", "r") as file:
    #         data = json.load(file)
    #     num_pairs = len(data)
    #     if num_pairs == 0 and len(new_value["decoder"]) == 0 and len(new_value["cross_attention"]) == 0:
    #         new_key = "encoder"
    #         data [new_key] = new_value["encoder"]
    #     else:
    #         new_key = "token_"+str(num_pairs-1)
    #         _ = new_value.pop("encoder")
    #         data[new_key] = new_value
    #     with open("tmp_files/all_stored_scaled_attentions.json", "w") as file:
    #         json.dump(data, file, indent=4)

    # def store_topk(self):
    #     new_value = self.topk
    #     with open("tmp_files/all_topk.json", "r") as file:
    #         data = json.load(file)
    #     num_pairs = len(data)
    #     new_key = "token_"+str(num_pairs)
    #     data[new_key] = new_value
    #     with open("tmp_files/all_topk.json", "w") as file:
    #         json.dump(data, file, indent=4)

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def generate(
        self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None, seed=0, average_across_batch=False, env=None
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """
        torch.manual_seed(seed)
        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        if self.use_two_hot:
            generated = src_len.new(max_len, bs).to(self.dtype)  # upcoming output
        else:
            generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        two_hot_constant_masks = [torch.zeros_like(generated[0], dtype=bool)]

        self.attn_stored_all_tokens = []
        self.attn_scaled_stored_all_tokens = []
        self.intermediate_tokens_all = []

        # cache compute states
        self.cache = {"slen": 0}
        # print(f"start loop inside generate")
        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            self.attn_stored_all_tokens.append(self.stored_attn_weights)
            self.attn_scaled_stored_all_tokens.append(self.stored_scaled_attn_weights)
            self.intermediate_tokens_all.append(self.intermediate_tokens)

            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)  # (bs, dim)  ##BE CAREFUL
            scores = self.proj(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                if average_across_batch:
                    scores = scores.mean(dim=0, keepdim=True).expand(bs, scores.size(1))
                logits = F.softmax(scores.float()/sample_temperature, dim=1)
                next_words = torch.multinomial(logits, num_samples=1).squeeze(1)
            assert next_words.size() == (bs,)

            if self.use_two_hot:
                next_words, two_hot_constant_mask = env.topk_decode_two_hot(
                    logits=scores, topk_idx=next_words, unfinished_sents=unfinished_sents,
                )
                two_hot_constant_masks.append(two_hot_constant_mask)

            # # Print the decoded token
            # for word in next_words:
            #     print(self.id2word.get(word.item(), "<UNK>"))

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        # print(f"generated: {generated[:7,:]}")
        # mask of shape (seq_len, bs) which tells which elements of the generated sequences
        # are constants which have been two-hot decoded
        two_hot_constant_masks = torch.stack(two_hot_constant_masks)
        return generated[:cur_len], gen_len, two_hot_constant_masks

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200, average_across_batch=False, env=None
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
        generated = generated.to(self.dtype)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        # positions
        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        self.cache = {"slen": 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            if self.apex:
                tensor = tensor.data[-1, :, :].to(self.dtype)  # (bs * beam_size, dim)
            else:
                tensor = tensor.data[
                    -1, :, :
                ]  # .to(soui elf.dtype)  # (bs * beam_size, dim)
            scores = self.proj(tensor)  # (bs * beam_size, n_words)
            if average_across_batch:
                scores = scores.mean(dim=0, keepdim=True).expand_as(scores)
            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                input=_scores, k=2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            if self.use_two_hot:
                raise NotImplementedError()
                # we cant just decode to float here as, see beam_id and word_id which are obtained by div and modulo
                # next_words = env.topk_decode_two_hot(logits=scores, topk_idx=next_words)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # def get_coeffs(s):
        #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
        #     poly = np.poly1d(roots, r=True)
        #     coeffs = list(poly.coefficients.astype(np.int64))
        #     return [c % 10 for c in coeffs], coeffs

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         hh = " ".join(self.id2word[x] for x in ww.tolist())
        #         print(f"{ss:+.4f} {hh}")
        #         # cc = get_coeffs(hh[4:])
        #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap,
        then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_len ** self.length_penalty
            )


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(
            top_k=top_k,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        top_k = min(
            max(self.top_k, self.min_tokens_to_keep), scores.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
