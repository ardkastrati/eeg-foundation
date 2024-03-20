import sys

sys.path.append("/home/maxihuber/eeg-foundation")

import src.models.mae_original as mae
from src.data.transforms import crop_spectrogram, load_channel_data, fft_256
import torch
import mne
import lightning
import matplotlib.pyplot as plt
import json
import numpy as np

# options

train = True
stor_path = ""
ckpt_path = "/itet-stor/maxihuber/net_scratch/ckpts/epoch34_mask30.ckpt"


c_loader = load_channel_data(precrop=False)
fft = fft_256(window_size=2.0, window_shift=0.125, cuda=True)

crop = crop_spectrogram(target_size=(64, 2048))


model = mae.MaskedAutoencoderViT(
    in_chans=1,
    img_size=(64, 2048),
    embed_dim=384,
    depth=12,
    num_heads=6,
    decoder_embed_dim=512,
    decoder_num_heads=16,
    mlp_ratio=4,
    decoder_mode=1,
)

checkpoint = torch.load(ckpt_path)

state_dict = checkpoint["state_dict"]
prefix_to_remove = "net."
new_state_dict = {
    key.replace(prefix_to_remove, ""): value for key, value in state_dict.items()
}
checkpoint["state_dict"] = new_state_dict

model.load_state_dict(checkpoint["state_dict"])

model.eval()
model.cuda()


if train:
    with open(
        "/home/maxihuber/eeg-foundation/indices_and_labels/tuab_train_labeled", "r"
    ) as file:
        tuab_train = json.load(file)
else:
    with open(
        "/home/maxihuber/eeg-foundation/indices_and_labels/tuab_eval_labeled", "r"
    ) as file:
        tuab_train = json.load(file)


path_prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"


train_emb = {}
print("starting the embed generation")
with torch.no_grad():
    for index, (path, lbl) in enumerate(tuab_train):

        print(index)

        path = path_prefix + path
        edf_file = mne.io.read_raw_edf(path, preload=False)
        channels = edf_file.ch_names
        cls_tokens = []
        for ch in channels:

            if not "EEG" in ch:
                continue
            ind = {"path": path, "chn": ch}
            channel_data = c_loader(ind)

            segments = []
            segments.append(channel_data[60 * 256 : 316 * 256])
            segments.append(channel_data[316 * 256 : 572 * 256])
            segments.append(channel_data[572 * 256 : 828 * 256])

            embeds = []

            for seg in segments:
                seg = seg.to("cuda")
                spg = fft(seg)
                spg = crop(spg)

                spg = spg.unsqueeze(0).unsqueeze(0)
                spg = spg.to("cuda")
                spg = model.forward_encoder_no_mask(spg)

                spg = spg.cpu().numpy().squeeze(0)

                spg = spg[0]

                spg = spg.tolist()
                embeds.append(spg)

            cls_tokens.append((ch, embeds))

        train_emb[path] = (cls_tokens, lbl)

print("saving embeddings")
with open(stor_path, "w") as file:
    json.dump(train_emb, file)
