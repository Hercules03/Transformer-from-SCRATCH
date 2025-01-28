This is a program that code a Transformer following the journal "Attention is all we need" from scratch. And this is actually a learning program from myself. I will put down my notes in between my notes and this READme files as well.

Pipeline
====================
1. Input Text:
    "What is your name?"

2. Tokenization:
    # The sentence gets split into tokens
    tokens = ["what", "is", "your", "name", "?"]

3. Token to IDs:
    # Each token gets converted to a numerical ID using a vocabulary
    token_ids = [456, 89, 234, 567, 45]

4. Embedding Layer:
    # Token IDs get converted to embeddings
    embeddings = embedding_layer(token_ids)
    # Shape: (1, 5, embed_size)  # 5 is sequence length

5. Positional Encoding:
    # Add positional information to embeddings
    embeddings = embeddings + positional_encoding

6. Encoder Processing:
    a) Self-Attention:
        # The same embedded sequence is used as Q, K, and V
        attention_output = self_attention(
            query=embeddings,    # (1, 5, embed_size)
            key=embeddings,      # (1, 5, embed_size)
            value=embeddings     # (1, 5, embed_size)
        )
    
    b) Add & Norm:
        # Residual connection and layer normalization
        output = layer_norm(attention_output + embeddings)
    
    c) Feed Forward:
        # Pass through feed-forward neural network
        ff_output = feed_forward(output)
    
    d) Add & Norm:
        # Another residual connection and normalization
        encoder_output = layer_norm(ff_output + output)

7. Decoder Processing:
    a) Target Embedding:
        # Embed the target sequence (during training or previous outputs during inference)
        target_embeddings = embedding_layer(target_tokens)
    
    b) Masked Self-Attention:
        # Self-attention with future masking
        masked_attention = masked_self_attention(
            target_embeddings,
            target_embeddings,
            target_embeddings,
            mask=triangular_mask
        )
    
    c) Cross-Attention:
        # Attention between decoder and encoder outputs
        cross_attention = attention(
            query=masked_attention,
            key=encoder_output,
            value=encoder_output
        )
    
    d) Feed Forward & Normalization:
        # Final processing
        decoder_output = feed_forward(cross_attention)

8. Final Linear Layer:
    # Project to vocabulary size
    logits = linear_layer(decoder_output)
    # Shape: (1, seq_len, vocab_size)

9. Softmax:
    # Convert to probabilities
    probabilities = softmax(logits)

10. Output Processing:
    # Convert highest probability tokens to text
    output_ids = argmax(probabilities)
    output_tokens = convert_ids_to_tokens(output_ids)
    final_text = join_tokens(output_tokens)




Decoder
===================
1. Input
2. Embedding Layer
3. Positional Encoding
4. Multi-Head Attention (QKV -> Attention -> Concat -> Linear)
    - Gather relevant information from other positions
5. LayerNorm
6. FFN
    - Process and transform this gathered information to extract useful features
7. LayerNorm