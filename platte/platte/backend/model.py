
import lightning as L
L.seed_everything(42)

#ckpt_path = '/itet-stor/maxihuber/net_scratch/checkpoints/980473/epoch=7-step=239317-val_loss=130.45-lr.ckpt'
#ckpt_path = '/itet-stor/maxihuber/net_scratch/checkpoints/977598/epoch=0-step=32807-val_loss=133.55.ckpt'

from backend.models.mae_rope_encoder import EncoderViTRoPE
from backend.models.components.vit_rope import (
    Flexible_RoPE_Layer_scale_init_Block,
    FlexibleRoPEAttention,
    compute_axial_cis,
    select_freqs_cis,
)
from timm.models.vision_transformer import Mlp as Mlp
from torch.nn import TransformerEncoderLayer

class SingleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(SingleTransformerEncoderLayer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead)

    def forward(self, src):
        return self.encoder_layer(src)

from src.models.components.SimpleTransformer import SimpleTransformer

def mean_aggregation(tokens):
    return torch.mean(torch.stack(tokens), dim=0)

from sklearn.metrics import balanced_accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchmetrics
from functools import partial


class FineTuningModel(L.LightningModule):
    def __init__(self, encoder, frozen_encoder, out_dim, task_name, task_type, learning_rate, mask_ratio):
        super(FineTuningModel, self).__init__()

        self.task_name = task_name
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio

        # Pretrained network
        self.encoder = encoder       
        if frozen_encoder:
            self.freeze_encoder()

        # Finetuning network
        self.finetune_time_transformer = Flexible_RoPE_Layer_scale_init_Block(
            dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            Attention_block=FlexibleRoPEAttention,
            Mlp_block=Mlp,
            init_values=1e-4,
        )
        
        # Single 1D transformer encoder layer
        #self.finetune_channel_transformer = SingleTransformerEncoderLayer(
        #    d_model=384,  # Match the dimension used in finetune_time_transformer
        #    nhead=1       # Number of heads in the multiheadattention models
        #)
        self.finetune_channel_transformer = SimpleTransformer(
            embed_size=384,
            max_len=8_500,
        )
        
        # Modular aggregation method on channel tokens
        self.win_shift_aggregation = mean_aggregation
        
        if task_type == "Regression":
            self.head = nn.Linear(encoder.encoder_embed_dim, out_dim)
            self.criterion = nn.MSELoss()
        else:
            self.head = nn.Linear(encoder.encoder_embed_dim, out_dim)
            self.criterion = nn.CrossEntropyLoss()

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, full_x):
        
        x_embeds = {}
        H_W = {}
        
        for win_size, x_win in full_x.items():
            spgs = x_win["batch"]
            channels = x_win["channels"]
            means = x_win["means"]
            stds = x_win["stds"]
            B, C, H, W = spgs.shape
            # TODO: split into less rows if necessary because of CUDA error
            #nr_tokens = B * C * H * W
            #if nr_tokens > max_nr_tokens:
            #    pass
            x_emb, _, _, nr_meta_patches = self.encoder(
                x=spgs,
                means=means,
                stds=stds,
                channels=channels,
                win_size=win_size,
                mask_ratio=self.mask_ratio,
            )
            # TODO: 
            x_embeds[win_size] = x_emb
            H_W[win_size] = (H, W)
            #print(f"[FT.forward, after self.encoder] x_emb.shape: {x_emb.shape}")

        # Pass through time-transformer
        for win_size, x_emb in x_embeds.items():
            freqs_cis = select_freqs_cis(
                self.encoder, self.encoder.encoder_freqs_cis, H_W[win_size][0], H_W[win_size][1], win_size, x_emb.device
            )
            x_emb = self.finetune_time_transformer(x_emb, freqs_cis=freqs_cis, nr_meta_tokens=nr_meta_patches)
            #print(f"[FT.forward, after self.time_transformer] x_emb.shape: {x_emb.shape}")
            x_emb = x_emb[:, 0]
            #print(f"[FT.forward, after time-token] x_emb.shape: {x_emb.shape}")
            x_embeds[win_size] = x_emb

        # Pass through channel-transformer
        tokens = []
        for win_size, x_emb in x_embeds.items():
            x_emb = x_emb.unsqueeze(0)
            #print(f"[FT.forward, before channel-token] x_emb.shape: {x_emb.shape}")
            x_emb = self.finetune_channel_transformer(x_emb)
            x_emb = x_emb[0, 0]
            #print(f"[FT.forward, after channel-token] x_emb.shape: {x_emb.shape}")
            tokens.append(x_emb)

        #print(f"[FT.forward] len(tokens): {len(tokens)}")
        # Average over all window shifts
        smart_token = self.win_shift_aggregation(tokens)
        #print(f"[FT.forward] smart_token.shape: {smart_token.shape}")

        # Pass through head
        y_hat = self.head(smart_token)
        
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y)
        self.log('train_loss', loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = torch.argmax(y_hat, dim=0)
            #print(f"[training_step] y_hat={y_hat}, y_pred={y_pred}, y={y}, loss={loss}")
            self.train_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.train_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))
        
        return loss

    def on_train_epoch_end(self):
        self.compute_metrics(self.train_step_outputs, 'train')
        self.train_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y)
        self.log('val_loss', loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = torch.argmax(y_hat, dim=0)
            #print(f"[validation_step] y_pred={y_pred}, y={y}, loss={loss}")
            self.validation_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.validation_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))

        return loss

    def on_validation_epoch_end(self):
        self.compute_metrics(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y)
        self.log('test_loss', loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = torch.argmax(y_hat, dim=0)
            #print(f"[test_step] y_pred={y_pred}, y={y}, loss={loss}")
            self.test_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.test_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))

        return loss

    def on_test_epoch_end(self):
        self.compute_metrics(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def compute_metrics(self, outputs, stage):
        y_true_all = defaultdict(list)
        y_pred_all = defaultdict(list)
        
        for y_true, y_pred, dataset in outputs:
            y_true_all[dataset].append(y_true)
            y_pred_all[dataset].append(y_pred)

        overall_y_true = []
        overall_y_pred = []

        for dataset in y_true_all.keys():
            y_true_cat = torch.stack(y_true_all[dataset])
            y_pred_cat = torch.stack(y_pred_all[dataset])

            overall_y_true.append(y_true_cat)
            overall_y_pred.append(y_pred_cat)

            if self.task_type == "Classification":
                balanced_acc = balanced_accuracy_score(y_true_cat, y_pred_cat)
                self.log(f'{stage}_balanced_accuracy_{dataset}', balanced_acc, prog_bar=True)
            elif self.task_type == "Regression":
                rmse_value = rmse(y_true_cat, y_pred_cat)
                self.log(f'{stage}_rmse_{dataset}', rmse_value, prog_bar=True)

        # Compute overall metrics
        overall_y_true = torch.cat(overall_y_true, dim=0)
        overall_y_pred = torch.cat(overall_y_pred, dim=0)

        if self.task_type == "Classification":
            balanced_acc = balanced_accuracy_score(overall_y_true, overall_y_pred)
            self.log(f'{stage}_balanced_accuracy', balanced_acc, prog_bar=True)
        elif self.task_type == "Regression":
            rmse_value = rmse(overall_y_true, overall_y_pred)
            self.log(f'{stage}_rmse', rmse_value, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)

    def on_train_epoch_start(self):
        if trainer.current_epoch == 1:
            self.unfreeze_encoder()
            print(f"Unfroze encoder at epoch {self.trainer.current_epoch}")
        
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    m = student()
    m.eval()

    model = student()
    checkpoint = t.load("../example.pth", map_location=t.device('cpu'))
    model.load_state_dict(checkpoint['student_model'])
    model.eval()

#    x2 = t.ones((1,3,192,256))
#    y = m(x)
#    y2 = m(x2)
#    y = y.detach().numpy()
#    y2 = y2.detach().numpy()

#    for i, prediction in enumerate(y[:, 0, :, :]):
#        img_data = post_process_png(prediction, (256,192))
#        print(img_data.shape)