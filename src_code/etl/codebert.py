import torch
from transformers import RobertaModel, RobertaTokenizer
import logging

logging.basicConfig(level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
EMBED_MODEL = RobertaModel.from_pretrained("microsoft/codebert-base")
EMBED_MODEL.to(device)
