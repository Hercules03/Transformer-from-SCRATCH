import torch
import torch.nn as nn

""" 
nn.Module
==========
1. Automatically keeps track of all the learnable parameters in the model
2. Organize nn components in a hierachical way
3. Layers automatically integrates with PyTorch's autograd system for backpropagation
4. Provide methods to manage state (train, eval, state_dict, to(device))
"""


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()  #inherut from base class (nn.Module)
        self.embed_size = embed_size
        self.heads = heads  # How many parallel attention operations will be performed
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads = embed_size), "Embed size needs to be divisible by heads"
        
        # Setup the QKV linear layer at the begining of the multi-head attention
        # Creating three different representations of the same input that serve distinct roles in attention calculation
        # nn.Linear(in_features, out_features, biase=True, device=None, dtype=None)
        # Two layers (input, output) of neurons fully connected to each other
        """ 
        Input Layer (64)          Output Layer (64)
        o ----------------------→ o
        o ----------------------→ o
        o ----------------------→ o
        o ----------------------→ o
        ...     4096 total     ...
        o       connections       o
        o ----------------------→ o
        o ----------------------→ o
        """
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)   # What information do I carry?
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # What do I contain
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)  # What am I looking for?
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)    # Stack those 8 ouptut from the heads into one output layer
    
    # Defines how data flows through the sel-attention mechanism
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Number of queries (Batch size) **Number of training exmaples are processed together in one forward pass
        # Each batch process N sentences at once
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]    # Typically it will be the length of input sequence
        # How many tokens/words/items are in each sequence in your batch
        
        # Reshape for multiple heads
        # Split embedding into self.heads pieces
        # .reshape(input, shape) -> Tensor
        # Before reshape:
        # values shape: (32, 100, 512)
        # where 512 is the full embedding dimension

        # After reshape:
        # values shape: (32, 100, 8, 64)
        # where 8 is number of heads and 64 is dimension per head
        """ 
        Why we need to do reshaping?
        =================================
        1. Enable Parallel Attention
            - Compute attention in parallel across multiple smaller subspaces
            - Each head can focus on different aspects
            - Capturing different types of relationships
        2. Dimension Splitting
            - Split the original embedding dimension (embed_size) into multiple heads
            - Each head works with a smaller dimension
        3. Independent Learning
            - Each head can learn to attend to different aspects of the input independently
            - One head might focus on syntactic relationships
            - Another might focus on semantic relationships
            - Multi-view representation -> Better model performance
            
        **Having multiple experts(heads) looking at the same input from different perspectives,
        rather than having a single expert lokking at everything at once.
        """
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        
        # Multiply the queries with the keys
        """ 
        EXAMPLE
        ============
        Input: "I love deep learning"
        When processing the word "deep", the energy scores tell you 
        how much this word should pay attention to:
            - "I" (maybe low attention score)
            - "love" (maybe medium attention score)
            - "deep" (high attention score)
            - "learning" (very high attention score)
        """
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys]) # a.k.a. Attention scores between queries and keys
        # Q * K^T
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        # For each word(query) in our target how much should we pay attention to each word(key) in our input
        
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))    # Set the weight to negative infinity to ignore those input
            
        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        
        # Taking the attention weights, multiplying these weight with the actual values
        # Creates a weighted sum of the values based on the attention scores
        # Collecting information from all values, weighted by how import each one is
                """ 
        EXAMPLE
        ============
        If we are processing the word "deep":
        - The attention weights might be [0.1, 0.2, 0.3, 0.4] for ["I", "love", "deep", "learning"]
        - Each value vector gets multipled by its corresponding weight
        - The results are summed to create the output representation for "deep"
        """
        out = torch.enisum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # value shape: (N, value_len, heads, heads_dim))
        # (N, query_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions
        
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        """ 
        Use of FFN
        ===============
        1. Introduce Non-linearity & Complexity
            - Self-attention is essentially a weighted sum operation (linear)
            - FNN adds non-linearity transformation through activation functions (usually ReLU)
        2. Feature Enhancement
            - Further processes the contextual relationships captured by the attention
        """
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU()
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(values, keys, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super().__init__()
        self.embed_size = embed_size    # It should be 512
        self.device = device
        
        # Store word embeddings and retrieve them using indices
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # Converts words(represented as indices) into dense vectors of fixed size
        self.position_embedding = nn.Embedding(max_length, embed_size)  # Adds positional information to the word embeddings
        
        # Creates a kist containing transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, mask):
            N, seq_length = x.shape
            positions = torch.arrange(0, seq_length).expand(N, seq_length).to(self.device)
            
            out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # Combines word and positional embeddings
            
            for layer in self.layers:
                out = layer(out, out, out, mask)    # out arguments represent query, key, and value matrices
            
            return out
        
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)   
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        
        
        """ 
        EXAMPLE (Machine Translation) English - French
        
        Source sequence
        ====================
        - Input sequence that needs to be transformed
        - English sentence
        - Processed be the ENCODER
        
        Target sequence
        ====================
        - Output sequence we want to generate
        - French sentence
        - Generated by DECODER
        
        """
        attention = self.attention(x, x, x, trg_mask)   # Masked Self-Attentionm uses trg_mask to prevent looking at the future tokens
        query = self.dropout(self.norm(attention + x))  # Residual connection
        
        # Cross-Attention
        out = self.transformer_block(value, key, query, src_mask)   # Uses the processed target sequence as queries, takes encoded source sequence information as keys and values
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size, 
                    heads, 
                    forward_expansion, 
                    dropout, 
                    device
                    ) for _ in range(num_layers)]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size) # Final layer that converts embeddings back to vocabulary probabilites
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, srx_mask, trg_mask):
        N, seq_length = x.shape # Get dimensions and positions
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)   # Creates position indices for each token
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions))) # Combine embeddings
        
        # Process through decoder blocks
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)  # enc_out for cross_attention
            
        out = self.fc_out(x)    # Prediction of which word is x
        
        return out
        
class Transformer(nn.Module):
    
    
    """
    Padding
    ============
    We need all sequences to have the same length since nns expected fixed-size inputs when processing sequence (like sentences) in a batch
    But naturally, sentences have different lengths. For example:
    
    Sentence 1: "I love cats" (3 words)
    Sentence 2: "The quick brown fox jumps" (5 words)
    
    To handle this, we "pad" shorter sequences with a special token (usually 0) to match the length of the longest sequences:
    
    Sentence 1: "I love cats <PAD> <PAD>"
    Sentence 2: "The quick brown fox jumps"
    
    Padding index (pad_idx)
    ============================
    - Special token (usually 0) used for padding
    - src_pad_idx: The padding token value for source sequences
    - trg_pad_idx: The padding token value for target sequences
    """
    def __init__(
        self,
        src_vocab_size, # Size of source vocabulary (e.g. Size of English vocab list)
        trg_vocab_size, # Size of target vocabulary (e.g. Size of French vocab list)
        src_pad_idx,    # Indices used for padding tokens
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg+pad_idx = trg_pad_idx
        self.device = device
        
    # Creates mask to ignore padding tokens in source sequence
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_length)
        return src_mask.to(self.device)
    
    # Create triangular mask for autoregressive generation
    # 1s allow attention to previous and current tokens
    # 0s prevent attention to future tokens
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        # Create masks for both source and target sequences
        src_mask = self.make_src_mask(src)  
        trg_mask = self.make_trg_mask(trg)
        
        # Encode the source sequence
        enc_src = self.encoder(src, src_mask)
        
        # Decode with target sequence and encoded source
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out