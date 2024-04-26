import numpy as np
import torch

def get_all_image_feats(dataloader, processor, get_image_features):
    """
    Gets image features from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_image_features: the model or model attribute used to
      extract image features
    Outputs:
    - a numpy array of shape [num_images, embed_dim]
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(images=d["images"], return_tensors="pt")
            feats = get_image_features(**inputs).detach().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)

def get_all_text_feats(dataloader, processor, get_text_features):
    """
    Gets text features from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_text_features: the model or model attribute used to
      extract text features
    Outputs:
    - a numpy array of shape [num_texts, embed_dim]
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(text=d["text"], return_tensors="pt", 
                              padding=True)
            feats = get_text_features(**inputs).detach().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)

def get_all_sim_scores(dataloader, processor, get_similarity_scores):
    """
    Gets image--text similarity scores from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_similarity_scores: the model or model attribute used to
      obtain similarity scores
    Outputs:
    - a numpy array of shape [num_trials, num_images_per_trial, 
      num_texts_per_trial]
    """
    all_sims = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(images=d["images"], text=d["text"], 
                              return_tensors="pt", padding=True)
            sims = get_similarity_scores(**inputs).detach().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)