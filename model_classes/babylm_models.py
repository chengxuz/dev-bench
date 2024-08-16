from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class BabyLMEvalModel(EvalModel):
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.image_processor = processor.image_processor
        if getattr(model, 'git', None) is not None:
            self.image_model = model.git.image_encoder
            self.need_image_prefix = False
        elif getattr(model, 'model', None) is not None:
            self.image_model = model.model.decoder.img_encoder
            self.need_image_prefix = True
        else:
            raise NotImplementedError
        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader using Bridge Tower model
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the BridgeTowerProcessor
        - model: the BridgeTowerModel
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                # Prepare inputs with padding and truncation
                # Assuming each data point in the dataloader has multiple images and texts
                num_images = len(d["images"])
                num_texts = len(d["text"])
                sims = np.zeros((num_images, num_texts))

                for i, image in enumerate(d["images"]):
                    #print(image)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    #print(image.mode)
                    scores = {}
                    for j, text in enumerate(d["text"]):
                        # Prepare inputs for each image-text pair
                        #inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
                        prompt = f"The caption for this image is: {text}."
                        if self.need_image_prefix:
                            prompt = '<image> ' + prompt
                        encoding = self.processor(images=image, text=prompt, return_tensors="pt")
                        encoding.pop('token_type_ids')
                        encoding['labels'] = encoding['input_ids']
                        # Forward pass
                        outputs = self.model(**encoding)
                        #import ipdb ; ipdb.set_trace()
                        scores[text] = -outputs['loss'].detach().cpu().numpy()
                        sims[i, j] = scores[text]

                # Append the similarity scores for this batch to all_sims
                all_sims.append(sims)
        
        return np.stack(all_sims, axis=0)
