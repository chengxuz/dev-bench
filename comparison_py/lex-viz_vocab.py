import os
from glob import glob
import numpy as np
import pandas as pd
from stats_helper import *

VV_DIR = "evals/lex-viz_vocab/"
human_data_vv = pd.read_csv(VV_DIR + "human.csv")

def compare_vv(model_data, human_data):
    kl_values = []
    beta_values = []
    iterations = []

    grouped = human_data.groupby('age_bin')
    for _, group in grouped:
        relevant_data = group.filter(regex='image|trial')
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

# comparisons for openclip
oc_files = glob(VV_DIR + "openclip/*.npy")
oc_kls = []
for ocf in oc_files:
    res = np.load(ocf)
    res = pd.DataFrame(res.squeeze(),
                       columns = ["image1", "image2", "image3", "image4"])
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = (res['image1'] > res['image2']) & \
                     (res['image1'] > res['image3']) & \
                     (res['image1'] > res['image4'])
    acc = res['correct'].mean()
    kls = compare_vv(res, human_data_vv)
    kls['epoch'] = int(os.path.splitext(os.path.basename(ocf))[0].replace("openclip_epoch_", ""))
    kls['accuracy'] = acc
    oc_kls.append(kls)
openclip_div_vv = pd.concat(oc_kls).sort_values(["epoch", "age_bin"]).reset_index(drop=True)
openclip_div_vv.to_csv("comparison/lex-vv_openclip.csv")

# comparisons for other models
vv_files = glob(VV_DIR + "*.npy") + [VV_DIR + "openclip/openclip_epoch_256.npy"]
all_kls = []
for vvf in vv_files:
    res = np.load(vvf)
    res = pd.DataFrame(res.squeeze(),
                       columns = ["image1", "image2", "image3", "image4"])
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = (res['image1'] > res['image2']) & \
                     (res['image1'] > res['image3']) & \
                     (res['image1'] > res['image4'])
    acc = res['correct'].mean()
    kls = compare_vv(res, human_data_vv)
    kls['model'] = os.path.splitext(os.path.basename(vvf))[0].replace("vv_", "").replace("_epoch_256", "")
    kls['accuracy'] = acc
    all_kls.append(kls)
other_res_vv = pd.concat(all_kls).sort_values(["model", "age_bin"]).reset_index(drop=True)
other_res_vv.to_csv("comparison/lex-vv_models.csv")