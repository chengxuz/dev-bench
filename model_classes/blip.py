from eval_model import EvalModel
from tqdm import tqdm
import torch

class BlipEvalModel(EvalModel):
    def __init__(self, model, processor=None, image_model=None, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.image_model = image_model

        self.get_image_features = self.model.get_image_features
        self.get_text_features = self.model.get_text_features
        self.get_similarity_scores = lambda **x: self.model(**x).logits_per_image

    def get_all_text_feats(self, data_loader):
        all_text_feats = []
        
        for batch in tqdm(data_loader, desc="Extracting text features"):
            text_features = self.model(batch["text"])
            all_text_feats.append(text_features)
        
        all_text_feats = torch.cat(all_text_feats, dim=0)
        return all_text_feats