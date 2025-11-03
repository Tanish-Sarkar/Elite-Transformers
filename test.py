# import torch
# print("PyTorch LIVE")
# print("GPU:", torch.cuda.is_available())

import torch
print("PyTorch + GPU:", torch.cuda.is_available())
text = "Hey brother, let’s crush Transformers"
tokens = text.lower().split()
vocab = {w:i for i,w in enumerate(tokens)}
emb = torch.nn.Embedding(len(vocab), 8)
print(emb(torch.tensor([0,1,2])))