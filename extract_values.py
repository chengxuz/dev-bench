import numpy as np

def get_all_image_feats(dataloader, processor, get_image_features):
    """
    Gets image features from a dataloader; expects trial_type=1
    -----
    Inputs:
    - dataloader: a dataloader constructed from a dataset with 
      trial_type=1
    - processor: the appropriate input processor for the model
    - get_image_features: the model or model attribute used to
      extract image features
    Outputs:
    - a numpy array of shape [num_images, embed_dim]
    """
    if dataloader.trial_type != 1:
        raise ValueError("Dataloader must have trial_type 1")
    all_feats = []
    for d in dataloader:
        inputs = processor(images=d["image1"], return_tensors="pt")
        feats = get_image_features(**inputs).detach().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)

def get_all_text_feats(dataloader, processor, get_text_features):
    """
    Gets text features from a dataloader; expects trial_type=0
    -----
    Inputs:
    - dataloader: a dataloader constructed from a dataset with 
      trial_type=0
    - processor: the appropriate input processor for the model
    - get_text_features: the model or model attribute used to
      extract text features
    Outputs:
    - a numpy array of shape [num_texts, embed_dim]
    """
    if dataloader.trial_type != 0:
        raise ValueError("Dataloader must have trial_type 0")
    all_feats = []
    for d in dataloader:
        inputs = processor(text=d["text1"], return_tensors="pt", 
                           padding=True)
        feats = get_text_features(**inputs).detach().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)

def get_all_sim_scores(dataloader, processor, get_similarity_scores):
    """
    Gets image--text similarity scores from a dataloader; expects 
    trial_type=2 or 4
    -----
    Inputs:
    - dataloader: a dataloader constructed from a dataset with 
      trial_type=2 or 4
    - processor: the appropriate input processor for the model
    - get_similarity_scores: the model or model attribute used to
      obtain similarity scores
    Outputs:
    - a numpy array of shape [num_trials, num_images_per_trial, 
      num_texts_per_trial]
    """
    if dataloader.trial_type != 2 and dataloader.trial_type != 4:
        raise ValueError("Dataloader must have trial_type 2 or 4")
    all_sims = []
    for d in dataloader:
        inputs = processor(images=d["images"], text=d["texts"], 
                           return_tensors="pt", padding=True)
        sims = get_similarity_scores(**inputs).detach().numpy()
        all_sims.append(sims)
    return np.stack(all_sims, axis=0)