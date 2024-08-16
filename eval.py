import data_handling
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluate a model on DevBench tasks.')
parser.add_argument('model', type=str, help='model type')
args = parser.parse_args()

# Load model
model_type = args.model # "clip_base"

if model_type == "clip_base":
    from model_classes.clip import ClipEvalModel
    from transformers import CLIPProcessor, CLIPModel

    eval_model = ClipEvalModel(
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), 
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    )

elif model_type == "clip_large":
    from model_classes.clip import ClipEvalModel
    from transformers import CLIPProcessor, CLIPModel

    eval_model = ClipEvalModel(
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14"), 
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    )
    
elif model_type == "blip":
    from model_classes.blip import BlipEvalModel
    from transformers import AutoProcessor, BlipForImageTextRetrieval, BlipModel

    eval_model = BlipEvalModel(
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco"), 
        image_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base"),
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    )

elif model_type == "flava":
    from model_classes.flava import FlavaEvalModel
    from transformers import FlavaProcessor, FlavaFeatureExtractor, FlavaForPreTraining, FlavaModel
    
    eval_model = FlavaEvalModel(
        model = FlavaForPreTraining.from_pretrained("facebook/flava-full"), 
        processor = FlavaProcessor.from_pretrained("facebook/flava-full"), 
        image_model = FlavaModel.from_pretrained("facebook/flava-full"), 
        feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
    )

elif model_type == "bridgetower":
    from model_classes.bridgetower import BridgetowerEvalModel
    from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval, BridgeTowerModel

    eval_model = BridgetowerEvalModel(
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm"),
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm"),
        image_processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base"),
        image_model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
    )

elif model_type == "vilt":
    from model_classes.vilt import ViltEvalModel
    from transformers import ViltProcessor, ViltForImageAndTextRetrieval, ViltModel

    eval_model = ViltEvalModel(
        model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco"),
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco"),
        vilt_base_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm"),
        vilt_base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    )

elif model_type == "cvcl":
    from model_classes.cvcl import CvclEvalModel
    from multimodal.multimodal_lit import MultiModalLitModel

    cvcl, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
    cvcl.eval()

    eval_model = CvclEvalModel(
        model = cvcl,
        processor = preprocess
    )

elif model_type == "siglip":
    from model_classes.siglip import SiglipEvalModel
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    print("loading siglip model")

    eval_model = SiglipEvalModel(
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384"),
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384"),
        model_embed = AutoModel.from_pretrained("google/siglip-base-patch16-224"),
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224"),
        processor_embed = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    )
    print("loaded siglip model")

elif model_type == "llava":
    from model_classes.llava import LlavaEvalModel
    from transformers import AutoProcessor, AutoModelForPreTraining
    eval_model = LlavaEvalModel(
            processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf"),
            model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf"))

elif model_type == "tinyllava":
    from transformers import AutoProcessor, AutoModelForPreTraining
    from model_classes.tinyllava import TinyLlavaEvalModel
    eval_model = TinyLlavaEvalModel(
            processor = AutoProcessor.from_pretrained("bczhou/tiny-llava-v1-hf"),
            model = AutoModelForPreTraining.from_pretrained("bczhou/tiny-llava-v1-hf"))

elif model_type in ["babylm_git_2024", "babylm_flamingo_2024"]:
    from model_classes.babylm_models import BabyLMEvalModel
    from transformers import AutoProcessor, AutoModelForCausalLM
    if model_type == "babylm_flamingo_2024":
        model_name = 'babylm/flamingo-2024'
    else:
        model_name = 'babylm/git-2024'

    eval_model = BabyLMEvalModel(
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True), 
        processor = AutoProcessor.from_pretrained(model_name),
    )

else:
    raise Exception(f"No implementation found for model '{model_type}'")



# Lexical tasks
lwl_ds = data_handling.DevBenchDataset("assets/lex-lwl/")
lwl_dl = data_handling.make_dataloader(lwl_ds)
print("getting all sim scores lwl")
lwl_sims = eval_model.get_all_sim_scores(lwl_dl)
np.save(f"evals/lex-lwl/lwl_{model_type}.npy", lwl_sims)

wg_ds = data_handling.DevBenchDataset("assets/gram-winoground/")
wg_dl = data_handling.make_dataloader(wg_ds)
wg_sims = eval_model.get_all_sim_scores(wg_dl)
np.save(f"evals/gram-winoground/wg_{model_type}.npy", wg_sims)

things_ds = data_handling.DevBenchDataset("assets/sem-things/")
things_dl = data_handling.make_dataloader(things_ds)
things_embeds = eval_model.get_all_image_feats(things_dl)
np.save(f"evals/sem-things/things_{model_type}.npy", things_embeds)

if False:
    vv_ds = data_handling.DevBenchDataset("assets/lex-viz_vocab/")
    vv_dl = data_handling.make_dataloader(vv_ds)
    vv_sims = eval_model.get_all_sim_scores(vv_dl)
    np.save(f"evals/lex-viz_vocab/vv_{model_type}.npy", vv_sims)

    # Grammatical tasks
    trog_ds = data_handling.DevBenchDataset("assets/gram-trog/")
    trog_dl = data_handling.make_dataloader(trog_ds)
    trog_sims = eval_model.get_all_sim_scores(trog_dl)
    np.save(f"evals/gram-trog/trog_{model_type}.npy", trog_sims)

    voc_ds = data_handling.DevBenchDataset("assets/sem-viz_obj_cat/")
    voc_dl = data_handling.make_dataloader(voc_ds)
    voc_embeds = eval_model.get_all_image_feats(voc_dl)
    np.save(f"evals/sem-viz_obj_cat/voc_{model_type}.npy", voc_embeds)

    wat_ds = data_handling.DevBenchDataset("assets/sem-wat/")
    wat_dl = data_handling.make_dataloader(wat_ds)
    wat_embeds = eval_model.get_all_text_feats(wat_dl)
    np.save(f"evals/sem-wat/wat_{model_type}.npy", wat_embeds)
