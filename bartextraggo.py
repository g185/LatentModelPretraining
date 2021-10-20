from transformers import AutoTokenizer, BartForConditionalGeneration, AutoConfig
import pytorch_lightning as pl
import torch

class bartextraggo_encoder(pl.LightningModule):
    