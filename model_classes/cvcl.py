from eval_model import EvalModel
import torch
import numpy as np

class CvclEvalModel(EvalModel):
    def get_all_image_feats(self, dataloader):
        """
        Gets image features from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the model used to extract image features
        - device: torch device (cuda or cpu)
        Outputs:
        - a numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        self.model.eval()
        with torch.no_grad():
            for d in dataloader:
                # Process the images with the processor
                images = [self.processor(img.convert("RGB")) for img in d["images"]]
                images = torch.stack(images).to(self.device)
                image_features = self.model.encode_image(images)
                #print(image_features.shape)
                #processed_inputs = processor(images=d["images"], return_tensors="pt")
                #pixel_values = processed_inputs["pixel_values"]
                
                # Get image features using model's encode_image method
                #image_features = model.encode_image(pixel_values).detach().numpy()
                
                # Append features to the list
                all_feats.append(image_features)
        
        return np.concatenate(all_feats, axis=0)

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader using the MultiModalLitModel
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the MultiModalLitModel
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in dataloader:
                # Process images
                images = [self.processor(img) for img in d["images"]]
                images = torch.stack(images).to(self.device)
                # image_features = self.model.encode_image(images)

                # Tokenize and encode texts
                texts, texts_len = self.model.tokenize(d["text"])
                texts, texts_len = texts.to(self.device), texts_len.to(self.device)
                # text_features = self.model.encode_text(texts, texts_len)

                # Get logits for image-text pairs
                logits_per_image, _ = self.model(images, texts, texts_len)
                sims = logits_per_image.detach().cpu().numpy()
                all_sims.append(sims)
        return np.stack(all_sims, axis=0)