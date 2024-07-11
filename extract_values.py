import numpy as np
import torch
from tqdm import tqdm

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



def get_all_text_feats_flava(dataloader, tokenizer, model):
    """
    Gets text features from a dataloader using the FLAVA model
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the FLAVA model

    Outputs:
    - a list of numpy arrays, where each array represents the text features for a data point
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            text_tokens = tokenizer(d["text"], return_tensors="pt", padding=True, max_length=77)
            #inputs = processor(text=d["text"], return_tensors="pt", padding=True)
            #inputs = {k: v.to(model.device) for k, v in inputs.items()}
            text_features = model.get_text_features(**text_tokens)[:, 0].float()
            #outputs = model(**inputs)
            #feats = outputs.text_embeddings.detach().cpu().numpy()
            all_feats.extend(text_features)
    return all_feats

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
            print(f"Similarity Scores Shape: {sims.shape}") 
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)

def get_all_sim_scores_gpu(dataloader, processor, get_similarity_scores):
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
        for d in tqdm(dataloader, desc="Processing data"):
            images_tensor = torch.stack(d["images"]).to('cuda')
            inputs = processor(images=images_tensor, text=d["text"], return_tensors="pt", max_length=77, padding=True, return_codebook_pixles=True, return_image_mask=True)
            sims = get_similarity_scores(**inputs).detach().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)
  
  
def get_all_sim_scores_flava(dataloader, processor, model):
    all_sims = []
    with torch.no_grad():
        for data in dataloader:
            # Convert all images to RGB
            images_rgb = [image.convert("RGB") for image in data["images"]]
            # Duplicate the text for each image to match the number of images
            texts = [data["text"][0]] * len(images_rgb)  # Ensure text count matches image count

            # Process the batch of images and texts
            inputs = processor(text=texts, images=images_rgb, return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True)
            outputs = model(**inputs)
        
            # Extract scores and reshape appropriately
            # Assuming the model outputs a score for each image-text pair and you only need the first score for each image
            scores = outputs.contrastive_logits_per_image[:, 0].view(-1, 1).numpy()  # Reshape to (2 images, 1 text)
            print(scores.shape)
            all_sims.append(scores)
    return np.array(all_sims)
  
  
def get_text_embeddings_blip(texts, processor, model):
    """
    Extracts text embeddings for a list of texts using a specified processor and model.
    
    Inputs:
    - texts: List of text strings
    - processor: Text processor that prepares input for the model
    - model: Model that provides the text embedding feature
    
    Outputs:
    - A tensor of shape [num_texts, embed_size] containing the text embeddings
    """
    with torch.no_grad():
        # Process the texts
        inputs = processor(text=texts, padding=True, return_tensors="pt")
        
        # Get text features from the model
        text_features = model.get_text_features(**inputs)
        
        # Assuming text_features is a tensor of shape [num_texts, embed_size]
        return text_features

        
    
    

""" def get_all_sim_scores_flava_trog(dataloader, processor, model):
    all_sims = []
    with torch.no_grad():
        for data in dataloader:
            images_rgb = [image.convert("RGB") for image in data["images"]]
            single_text = data["text"][0]  # Assuming there is one text for all images in the trial
            
            # Process all images with the same single text at once
            inputs = processor(text=[single_text] * len(images_rgb), images=images_rgb, return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True)
            
            outputs = model(**inputs)
            # Assuming the model outputs a score for each image-text pair
            scores = outputs.contrastive_logits_per_image.view(-1, 1).numpy()  # Reshape to (num_images, 1)
            
            all_sims.append(scores)

    return np.array(all_sims)  # Final shape will be (num_trials, num_images_per_trial, 1)  """





import torch
import numpy as np

def get_all_sim_scores_bridgetower(dataloader, processor, model):
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
        for d in dataloader:
            # Prepare inputs with padding and truncation
            # Assuming each data point in the dataloader has multiple images and texts
            num_images = len(d["images"])
            num_texts = len(d["text"])
            sims = np.zeros((num_images, num_texts))
            images_rgb = [image.convert("RGB") for image in d["images"]]
            for i, image in enumerate(images_rgb):
                for j, text in enumerate(d["text"]):
                    # Prepare inputs for each image-text pair
                    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
                    # Forward pass
                    outputs = model(**inputs)
                    sims[i, j] = outputs.logits[0, 1].item()

            # Append the similarity scores for this batch to all_sims
            all_sims.append(sims)
    
    return np.stack(all_sims, axis=0)


def get_all_sim_scores_flava_trog(dataloader, processor, model):
    all_sims = []
    with torch.no_grad():
        for data in dataloader:
            # Convert all images to RGB and process them together
            images_rgb = [image.convert("RGB") for image in data["images"]]
            single_text = data["text"][0]  # Assuming there is one text for all images in the trial

            # Create a batch with the same text repeated for each image
            repeated_texts = [single_text] * len(images_rgb)

            # Process all images and the repeated text in one go
            inputs = processor(text=repeated_texts, images=images_rgb, return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True)
            
            outputs = model(**inputs)
            # Extract the logits and assume the relevant score is at a specific index, e.g., [0, 1]
            scores = outputs.contrastive_logits_per_image[:, 1].view(-1, 1).numpy()  # Reshape to (num_images, 1)
            print(scores)
            all_sims.append(scores)

    return np.array(all_sims)  # Ensures the output shape is [num_trials, 4, 1]



from PIL import Image

def get_all_text_feats_bridgetower(dataloader, processor, model):
    """
    Gets text features from a dataloader using a model that outputs text features, preserving the feature dimension.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the model used to extract text features
    Outputs:
    - a numpy array of shape [num_texts, 768] where each row is a 768-dimensional vector representing the text
    """
    blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(images=blank_image, text=d["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            feats = outputs.text_features
            # Average across the sequence dimension only
            pooled_feats = feats.mean(dim=1).detach().numpy()  # This collapses the sequence dimension
            all_feats.append(pooled_feats)  # Append since we're reducing to 2D: (batch_size, 768)

    # Concatenate all batch features into a single array
    all_feats_array = np.concatenate(all_feats, axis=0)
    return all_feats_array
            


def get_all_sim_scores_cvcl(dataloader, preprocess, model):
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
            images_rgb = [image.convert("RGB") for image in d["images"]]
            images = [preprocess(img) for img in images_rgb]
            images = torch.stack(images).to(model.device)
            image_features = model.encode_image(images)

            # Tokenize and encode texts
            texts, texts_len = model.tokenize(d["text"])
            texts, texts_len = texts.to(model.device), texts_len.to(model.device)
            text_features = model.encode_text(texts, texts_len)

            # Get logits for image-text pairs
            logits_per_image, logits_per_text = model(images, texts, texts_len)
            sims = logits_per_image.detach().cpu().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)

def get_all_sim_scores_vilt(dataloader, processor, model):
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
        for d in dataloader:
            # Prepare inputs with padding and truncation
            # Assuming each data point in the dataloader has multiple images and texts
            num_images = len(d["images"])
            num_texts = len(d["text"])
            sims = np.zeros((num_images, num_texts))
            images_rgb = [image.convert("RGB") for image in d["images"]]

            for i, image in enumerate(images_rgb):
                #print(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                #print(image.mode)
                scores = {}
                for j, text in enumerate(d["text"]):
                    # Prepare inputs for each image-text pair
                    #inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
                    encoding = processor(image, text, return_tensors="pt")
                    # Forward pass
                    outputs = model(**encoding)
                    scores[text] = outputs.logits[0, :].item()
                    sims[i, j] = scores[text]

            # Append the similarity scores for this batch to all_sims
            all_sims.append(sims)
    
    return np.stack(all_sims, axis=0)

from tqdm import tqdm 

from PIL import Image

import torch
import numpy as np

def get_all_image_feats_bridgewater(dataloader, processor, model):
    """
    Gets image embeddings from a dataloader using a model that outputs embeddings.
    
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the model used to extract image embeddings
    
    Outputs:
    - a numpy array of shape [num_images, embed_dim]
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            # Process each image individually
            for image in d["images"]:
                # Convert image to RGB if not already in that format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Pass a blank text to satisfy model requirements
                inputs = processor(images=[image], text=[""], return_tensors="pt", padding=True, truncation=True)
                
                # Model inference
                outputs = model(**inputs)
                
                # Extract image features
                image_features = outputs.image_features  # Shape: (batch_size, image_sequence_length, hidden_size)
                
                # Average pooling over the sequence length dimension to get (batch_size, hidden_size)
                pooled_feats = image_features.mean(dim=1).squeeze().detach().numpy()
                #print(pooled_feats.shape)
                # Ensure the feature dimension is as expected and add to the list
                if len(pooled_feats.shape) == 1:  # When batch_size is 1
                    all_feats.append(pooled_feats)
                elif len(pooled_feats.shape) == 2:  # General case
                    all_feats.extend(pooled_feats)
                else:
                    print(f"Unexpected shape of pooled features: {pooled_feats.shape}")
    
    return np.array(all_feats)


def get_all_image_feats_flava(dataloader, fe, model):
    """
    Gets image features from a dataloader and applies mean pooling to each set of image features.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - fe: the feature extractor
    - model: the model used to extract image features
    Outputs:
    - a numpy array of shape [num_images, embed_dim] after mean pooling
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            images_rgb = [image.convert("RGB") for image in d["images"]]
            image_input = fe(images_rgb, return_tensors="pt")
            feats = model.get_image_features(**image_input).detach().numpy()
            mean_feats = np.mean(feats, axis=1)  # Mean pooling across the patches
            all_feats.append(mean_feats)
    return np.concatenate(all_feats, axis=0)



def get_all_image_feats_cvcl(dataloader, preprocess, model):
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
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            # Process the images with the processor
            images = [preprocess(img.convert("RGB")) for img in d["images"]]
            images = torch.stack(images).to(model.device)
            image_features = model.encode_image(images)
            print(image_features.shape)
            #processed_inputs = processor(images=d["images"], return_tensors="pt")
            #pixel_values = processed_inputs["pixel_values"]
            
            # Get image features using model's encode_image method
            #image_features = model.encode_image(pixel_values).detach().numpy()
            
            # Append features to the list
            all_feats.append(image_features)
    
    return np.concatenate(all_feats, axis=0)


def get_all_sim_scores_blip_trog(dataloader, processor, get_similarity_scores):
    all_sims = []
    with torch.no_grad():
        for d in dataloader:
            sims = []
            for image in d["images"]:
                inputs = processor(images=[image], text=[d["text"][0]], 
                                   return_tensors="pt", padding=True)
                sim_score = get_similarity_scores(**inputs).detach().numpy()
                # Squeeze unnecessary dimensions
                sims.append(sim_score)
            print(f"Similarity Scores for trial: {sims}") 
            all_sims.append(sims)
    return np.array(all_sims)




def get_all_text_feats_cvcl(texts, preprocess, model):
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
    model.eval()
    with torch.no_grad():
        texts, texts_len = model.tokenize(texts)
        #texts, texts_len = texts.to(device), texts_len.to(device)
        texts_features = model.encode_text(texts, texts_len)
       
            #all_feats.append(image_features)
    
    return texts_features



def get_text_embeddings_vilt(texts, processor, model):
    """
    Extracts text embeddings from a list of texts by passing each text to a CVCL model.
    -----
    Inputs:
    - texts: List of text strings
    - processor: the text processor that prepares input for the model
    - model: the CVCL model used to extract text embeddings
    Outputs:
    - A tensor of shape [num_texts, embed_dim] containing the text embeddings
    """
    embeddings = []
    blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
    for text in texts:
    # Prepare inputs without the image
        encoding = processor(images = blank_image, text=text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**encoding).pooler_output
        embeddings.append(outputs)
    return embeddings




import torch
import numpy as np
from PIL import Image


def get_all_sim_scores_generation(dataloader, model, tokenizer):
    """
    Gets image--text similarity scores from a dataloader using a specific prompt format.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - model: the model used to obtain similarity scores
    - tokenizer: the tokenizer used to prepare text inputs
    Outputs:
    - a numpy array of shape [num_trials, num_images * num_texts, 2] where the last dimension contains logits for "yes" and "no"
    """
    all_sims = []
    with torch.no_grad():
        for d in dataloader:
            trial_sims = []
            for image_path in d["images"]:
                enc_image = model.encode_image(image_path)
                #print(enc_image)
                for text in d['text']:
                    prompt = f"Image: <image>. Caption: {text}. Does the caption match the image? Answer either Yes or No."
                    #print(text)
                    inputs_embeds = model.input_embeds(prompt, enc_image, tokenizer)
                    #print(inputs_embeds)

                    # Generate with explicit attention mask and pad token settings
                    attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)  # Assuming inputs_embeds is 2D
                    outputs = model.text_model(
                        inputs_embeds=inputs_embeds,
                        #attention_mask=attention_mask,
                        #pad_token_id=tokenizer.eos_token_id,
                        output_hidden_states=True,
                        #return_dict_in_generate=True,
                        #max_new_tokens=1,  # Adjust as necessary to generate enough tokens
                    )
                
                    
                    logits = outputs.logits[:, -1, :]

                    # Use the exact tokens as they appear in the tokenizer's vocabulary
                    yes_token_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]
                    no_token_id = tokenizer(" No", add_special_tokens=False).input_ids[0]

                    # Pass each hidden state through the final linear layer to get logits
                    #logits = model.text_model.lm_head(hidden_states)

                    # Get logits for "yes" and "no"
                    yes_logits = logits[:,yes_token_id].squeeze()
                    no_logits = logits[:, no_token_id].squeeze()

        

                    # Stack the logits for "yes" and "no" to form the required output shape
                    pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                    #print(enc_image, text, pair_logits)
                    print(pair_logits)
                    trial_sims.append(pair_logits)
                    print(pair_logits.shape)

            all_sims.append(np.array(trial_sims))


    return np.array(all_sims)



def get_all_sim_scores_generation_gemma(dataloader, model, processor):
    """
    Gets image--text similarity scores from a dataloader using a specific prompt format.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - model: the model used to obtain similarity scores
    - tokenizer: the tokenizer used to prepare text inputs
    Outputs:
    - a numpy array of shape [num_trials, num_images * num_texts, 2] where the last dimension contains logits for "yes" and "no"
    """
    all_sims = []
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            trial_sims = []
            for image in d["images"]:
                #enc_image = model.encode_image(image_path)
                #print(enc_image)
                image = image.convert('RGB')
                for text in d['text']:
                    prompt = f"Caption: {text}. Does the caption match the image? Answer either Yes or No."
                    #print(text)
                    #inputs_embeds = model.input_embeds(prompt, image_path, tokenizer)
                    model_inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True, truncation=True)
                    #print(inputs_embeds)

                    # Generate with explicit attention mask and pad token settings
                    #attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)  # Assuming inputs_embeds is 2D
                    outputs = model(
                        input_ids=model_inputs['input_ids'],
                        attention_mask=model_inputs['attention_mask'],
                        #max_length=300,  # Adjust this based on your requirement
                        #max_new_tokens=100,  # Number of new tokens to generate
                        #do_sample=False
                    )
                    #print(outputs)
                
                    
                    #print(outputs.logits)
                    #print(outputs.logits.shape)
                    logits = outputs.logits[:, -1, :]

                    # Use the exact tokens as they appear in the tokenizer's vocabulary
                    yes_token_id = processor.tokenizer(" Yes", add_special_tokens=False).input_ids[0]
                    no_token_id = processor.tokenizer(" No", add_special_tokens=False).input_ids[0]

                    # Pass each hidden state through the final linear layer to get logits
                    #logits = model.text_model.lm_head(hidden_states)

                    # Get logits for "yes" and "no"
                    yes_logits = logits[:,yes_token_id].squeeze()
                    no_logits = logits[:, no_token_id].squeeze()

        

                    # Stack the logits for "yes" and "no" to form the required output shape
                    pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                    #print(enc_image, text, pair_logits)
                    print(f"Text: {text}, Yes logits: {yes_logits}, No logits: {no_logits}")
                    trial_sims.append(pair_logits)
                    print(pair_logits.shape)

            all_sims.append(np.array(trial_sims))


    return np.array(all_sims)



def get_image_embeds_generation(dataloader, model, tokenizer):
    """
    Gets image--text similarity scores from a dataloader using a specific prompt format.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - model: the model used to obtain similarity scores
    - tokenizer: the tokenizer used to prepare text inputs
    Outputs:
    - a numpy array of shape [num_trials, num_images * num_texts, 2] where the last dimension contains logits for "yes" and "no"
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            for image_path in d["images"]:
                enc_image = model.encode_image(image_path)
                if len(enc_image.shape) > 2:
                    # If enc_image has more than 2 dimensions, apply mean pooling
                    enc_image = enc_image.mean(dim=1)
                all_feats.append(enc_image)
                print(enc_image.shape)
    return np.concatenate(all_feats, axis=0)

         



def get_text_embeddings_generation(dataloader, model, tokenizer):
    """
    Gets image--text similarity scores from a dataloader using a specific prompt format.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - model: the model used to obtain similarity scores
    - tokenizer: the tokenizer used to prepare text inputs
    Outputs:
    - a numpy array of shape [num_trials, num_images * num_texts, 2] where the last dimension contains logits for "yes" and "no"
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            for text in d['text']:
                inputs_embeds = model.input_embeds(text, None, tokenizer)
                all_feats.append(inputs_embeds)
                print(inputs_embeds.shape)
                print(inputs_embeds)
    return np.concatenate(all_feats, axis=0)

                  