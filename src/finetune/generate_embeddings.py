import sys
sys.path.append("/home/maxihuber/eeg-foundation")

import src.models.mae_original as mae
from src.data.monai_transforms import crop_spectrogram, load_channel_data, fft_256
import torch
import mne
import lightning
import matplotlib.pyplot as plt
import json
import numpy as np
ckpt_path = "/home/maxihuber/long_run1.ckpt"


def plot_and_save(save_dir, image, save_local = True, save_log = False, log_tag =""):

        plt.pcolormesh(image, shading='auto', cmap='viridis')
        plt.ylabel('Frequency Bins')
        plt.xlabel('steps')
        plt.title('Spectrogram')
        plt.colorbar(label='')
        if save_local: 
                plt.savefig(save_dir)
        plt.clf()

test_path = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf/007/aaaaabdo/s003_2012_04_06/01_tcp_ar/aaaaabdo_s003_t000.edf"

c_loader = load_channel_data()
fft = fft_256()
crop = crop_spectrogram()

model = mae.MaskedAutoencoderViT(in_chans = 1,
embed_dim = 384,
depth = 12,
num_heads = 6,
decoder_embed_dim = 512,
decoder_num_heads = 16,
mlp_ratio = 4,
decoder_mode = 1)

checkpoint = torch.load(ckpt_path)

state_dict = checkpoint['state_dict']
prefix_to_remove = 'net.'
new_state_dict = {key.replace(prefix_to_remove, ''): value for key, value in state_dict.items()}
checkpoint['state_dict'] = new_state_dict

model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.cuda()

with open("/home/maxihuber/tuab_eval_labeled", 'r') as file:
        tuab_train = json.load(file)

print("opening /home/maxihuber/tuab_eval_labeled")

path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"


train_emb = {}
print("starting the embed generation")
with torch.no_grad(): 
        for index, (path, lbl) in enumerate(tuab_train):
                
                
                print (index)
               
                path = path_prefix + path
                edf_file = mne.io.read_raw_edf(path, preload=False)
                channels = edf_file.ch_names
                cls_tokens = []
                for ch in channels:
                        print(ch)
                        if not "EEG" in ch:
                                continue
                        ind = {"path" : path, "chn" : ch}
                        channel_data = c_loader(ind)
                        spg = fft(channel_data)
                        spg = crop(spg)
                        spg = spg.unsqueeze(0).unsqueeze(0)
                        spg = spg.to('cuda')
                        spg = model.forward_encoder_no_mask(spg)
                        
                        spg = spg.cpu().numpy().squeeze(0)
                        
                        spg = spg[0]
                        
                        spg = spg.tolist()
                        cls_tokens.append((ch, spg))

                train_emb[path] = (cls_tokens, lbl)
                
print("saving embeddings")
with open("tuab_eval_fine_train_emb", 'w') as file:
        json.dump(train_emb, file)
