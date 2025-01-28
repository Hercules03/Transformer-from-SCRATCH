import torch
import torch.nn as nn
from Transformer import Transformer

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define hyperparameters
    src_vocab_size = 1000
    trg_vocab_size = 1000
    src_pad_idx = 0
    trg_pad_idx = 0
    embed_size = 512
    num_layers = 6
    heads = 8
    dropout = 0.1
    max_length = 100
    forward_expansion = 4
    batch_size = 1
    src_seq_length = 6
    trg_seq_length = 5

    # Initialize model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        device=device,
        max_length=max_length,
        forward_expansion=forward_expansion
    ).to(device)

    # For translation: "Hello how are you" -> "Bonjour comment allez-vous"
    src = torch.tensor([[1, 2, 3, 4, 0, 0]])  # "Hello how are you" + padding
    trg = torch.tensor([[1, 2, 3, 4, 5]])     # "Bonjour comment allez-vous"

    # Test forward pass
    try:
        output = model(src, trg)
        print("Model forward pass successful!")
        print(f"Input shape (src): {src.shape}")
        print(f"Input shape (trg): {trg.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: (batch_size={batch_size}, trg_seq_length={trg_seq_length}, trg_vocab_size={trg_vocab_size})")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
