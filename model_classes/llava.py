from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class LlavaEvalModel(EvalModel):
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                trial_sims = []
                for image in d["images"]:
                    #enc_image = model.encode_image(image_path)
                    #print(enc_image)
                    for text in d['text']:
                        prompt = f"[INST] <image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No. [/INST]"
                        #print(text)
                        #inputs_embeds = model.input_embeds(prompt, enc_image, tokenizer)
                        inputs = self.processor(text = prompt, images = image, return_tensors = 'pt')
                        #print(inputs_embeds)

                        # Generate with explicit attention mask and pad token settings
                        #attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)  # Assuming inputs_embeds is 2D
                        logits = self.model(**inputs).logits.squeeze()
                        print(logits.shape)
                        # Use the exact tokens as they appear in the tokenizer's vocabulary
                        yes_token_id = self.processor.tokenizer.encode("Yes")[1]
                        no_token_id = self.processor.tokenizer.encode("No")[1]
                        print(yes_token_id)
                        print(no_token_id)
                        print(np.argmax(logits[-1]))
                        
                        yes_logits = logits[-1,yes_token_id]
                        no_logits = logits[-1, no_token_id]

            

                        # Stack the logits for "yes" and "no" to form the required output shape
                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                        #print(enc_image, text, pair_logits)
                        print(pair_logits)
                        trial_sims.append(pair_logits)
                        print(pair_logits.shape)
                        

                all_sims.append(np.array(trial_sims))


        return np.array(all_sims)
    

