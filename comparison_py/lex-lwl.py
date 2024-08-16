import os
from glob import glob
import numpy as np
import pandas as pd
from stats_helper import *

LWL_DIR = "evals/lex-lwl/"
human_data_lwl = pd.read_csv(LWL_DIR + "human.csv")

def compare_lwl(model_data, human_data):
    kl_values = []
    beta_values = []
    iterations = []

    grouped = human_data.groupby('age_bin')
    for _, group in grouped:
        relevant_data = group
        relevant_data['image1'] = relevant_data['prop']
        relevant_data['image2'] = 1. - relevant_data['image1']
        opt_kl = get_opt_kl(relevant_data, model_data)
        kl_values.append(opt_kl['objective'])
        beta_values.append(opt_kl['solution'])
        iterations.append(opt_kl['iterations'])
    
    result_df = pd.DataFrame({
        'age_bin': grouped.groups.keys(),
        'kl': kl_values,
        'beta': beta_values,
        'iterations': iterations
    })
    return result_df

if False:
    # comparisons for openclip
    oc_files = glob(LWL_DIR + "openclip/*.npy")
    oc_kls = []
    for ocf in oc_files:
        res = np.load(ocf)
        res = pd.DataFrame(res.squeeze(),
                           columns = ["image1", "image2"])
        res['trial'] = np.arange(res.shape[0])+1
        res['correct'] = (res['image1'] > res['image2'])
        acc = res['correct'].mean()
        kls = compare_lwl(res, human_data_lwl)
        kls['epoch'] = int(os.path.splitext(os.path.basename(ocf))[0].replace("openclip_epoch_", ""))
        kls['accuracy'] = acc
        oc_kls.append(kls)
    openclip_div_lwl = pd.concat(oc_kls).sort_values(["epoch", "age_bin"]).reset_index(drop=True)
    openclip_div_lwl.to_csv("comparison/lex-lwl_openclip.csv")

# comparisons for other models
lwl_files = glob(LWL_DIR + "*.npy") + [LWL_DIR + "openclip/openclip_epoch_256.npy"]
all_kls = []
for lwlf in lwl_files:
    if not os.path.exists(lwlf):
        continue
    print(lwlf)
    res = np.load(lwlf)
    res = pd.DataFrame(res.squeeze(),
                       columns = ["image1", "image2"])
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = (res['image1'] > res['image2'])
    acc = res['correct'].mean()
    kls = compare_lwl(res, human_data_lwl)
    kls['model'] = os.path.splitext(os.path.basename(lwlf))[0].replace("lwl_", "").replace("_epoch_256", "")
    kls['accuracy'] = acc
    print(kls)
    all_kls.append(kls)
other_res_lwl = pd.concat(all_kls).sort_values(["model", "age_bin"]).reset_index(drop=True)
other_res_lwl.to_csv("comparison/lex-lwl_models.csv")
