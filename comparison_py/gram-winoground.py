import os
import re
import io
from glob import glob
import numpy as np
import pandas as pd
from stats_helper import *

WG_DIR = "evals/gram-winoground/"

def get_human_data_wg(manifest_file="assets/gram-winoground/manifest.csv",
                      data_file="evals/gram-winoground/human.jsonl"):
    manifest_df = pd.read_csv(manifest_file)
    
    manifest_df['trial'] = manifest_df['image1'].str.extract(r"ex_([0-9]+)_img").astype(int) + 1
    included_trials = manifest_df['trial'].tolist()
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    processed_lines = [re.sub(r'[{}"]', '', line).replace('label: ', '').replace('score: ', '').strip() for line in lines]
    human_data = pd.read_csv(io.StringIO('\n'.join(processed_lines)), names=["label", "score"])
    
    human_data[['trial', 'text', 'image']] = human_data['label'].str.split('_', expand=True)
    human_data['trial'] = human_data['trial'].astype(int) + 1
    human_data['image'] = human_data['image'].str[1:].astype(int) + 1
    human_data['text'] = human_data['text'].str[1:].astype(int) + 1
    human_data['pair'] = human_data.apply(lambda row: f"image{row['image']}text{row['text']}", axis=1)
    human_data = human_data[human_data['trial'].isin(included_trials)]
    human_data['trial'] = human_data['trial'].apply(lambda t: included_trials.index(t) + 1)
    
    return human_data[['trial', 'pair', 'score']]

human_data_wg = get_human_data_wg()

def compare_wg(model_data, human_data):
    # process human data
    human_data_by_text = human_data.copy()
    human_data_by_text[['image', 'text']] = human_data_by_text['pair'].str.extract(r'(image[12])(text[12])')
    human_data_by_text = human_data_by_text.pivot(index=['trial', 'text'], columns='image', values='score').reset_index()
    human_data_by_text['rowsum'] = human_data_by_text.filter(like='image').sum(axis=1)
    for col in human_data_by_text.filter(like='image').columns:
        human_data_by_text[col] = human_data_by_text[col] / human_data_by_text['rowsum']
    human_data_by_text['trial'] = human_data_by_text.apply(lambda row: f"{row['trial']}_{row['text']}", axis=1)
    human_data_by_text = human_data_by_text.drop(columns=['rowsum', 'text'])
    
    # process model_data
    model_data_by_text = model_data.melt(id_vars=['trial'], value_vars=model_data.filter(like='image').columns,
                                         var_name='image_text', value_name='score')
    model_data_by_text[['image', 'text']] = model_data_by_text['image_text'].str.extract(r'(image[12])(text[12])')
    model_data_by_text = model_data_by_text.pivot(index=['trial', 'text'], columns='image', values='score').reset_index()
    model_data_by_text['trial'] = model_data_by_text.apply(lambda row: f"{row['trial']}_{row['text']}", axis=1)
    model_data_by_text = model_data_by_text.drop(columns=['text'])

    opt_kl = get_opt_kl(human_data_by_text, model_data_by_text)    
    result_df = pd.DataFrame({
        'kl': opt_kl['objective'],
        'beta': opt_kl['solution'],
        'iterations': opt_kl['iterations']
    }, index=[0])
    return result_df

# comparisons for openclip
oc_files = glob(WG_DIR + "openclip/*.npy")
oc_kls = []
for ocf in oc_files:
    if not os.path.exists(ocf):
        continue
    res = np.load(ocf)
    res = pd.DataFrame(res.reshape((-1, 4)),
                       columns = ["image1text1", "image1text2", "image2text1", "image2text2"]) # order is different than R
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = ((res['image1text1'] > res['image1text2']).astype(int) + \
                      (res['image2text2'] > res['image2text1']).astype(int))/2
    acc = res['correct'].mean()
    kls = compare_wg(res, human_data_wg)
    kls['epoch'] = int(os.path.splitext(os.path.basename(ocf))[0].replace("openclip_epoch_", ""))
    kls['accuracy'] = acc
    oc_kls.append(kls)
if len(oc_kls) > 0:
    openclip_div_wg = pd.concat(oc_kls).sort_values(["epoch"]).reset_index(drop=True)
    openclip_div_wg.to_csv("comparison/gram-wg_openclip.csv")

# comparisons for other models
wg_files = glob(WG_DIR + "*.npy") + [WG_DIR + "openclip/openclip_epoch_256.npy"]
all_kls = []
for wgf in wg_files:
    if not os.path.exists(wgf):
        continue
    res = np.load(wgf)
    res = pd.DataFrame(res.reshape((-1, 4)),
                       columns = ["image1text1", "image1text2", "image2text1", "image2text2"]) # order is different than R
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = ((res['image1text1'] > res['image1text2']).astype(int) + \
                      (res['image2text2'] > res['image2text1']).astype(int))/2
    acc = res['correct'].mean()
    kls = compare_wg(res, human_data_wg)
    kls['model'] = os.path.splitext(os.path.basename(wgf))[0].replace("wg_", "").replace("_epoch_256", "")
    kls['accuracy'] = acc
    print(kls)
    all_kls.append(kls)
other_res_wg = pd.concat(all_kls).sort_values(["model"]).reset_index(drop=True)
other_res_wg.to_csv("comparison/gram-wg_models.csv")
