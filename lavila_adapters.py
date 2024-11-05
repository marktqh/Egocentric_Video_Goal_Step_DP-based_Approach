import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, GPT2LMHeadModel

import lavila.lavila.models.loss as loss
from lavila.lavila.models.gpt2_gated import GPT2LMHeadModel as GatedGPT2LMHeadModel
from lavila.lavila.models.gpt2_gated import augment_gpt2_config
from lavila.lavila.models.narrator import VCLM_HF
from lavila.lavila.models.openai_clip import load as load_openai_clip
from lavila.lavila.models.openai_model import QuickGELU, Transformer
from lavila.lavila.models.timesformer import SpaceTimeTransformer, Adapter
from lavila.lavila.models.utils import remap_keys, rsetattr

class CLIP_HF(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 text_width: int,
                 text_model: nn.Module,
                 text_use_cls_token: bool,
                 text_is_regressive: bool,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model
        self.text_width = text_width
        self.textual = text_model
        self.text_use_cls_token = text_use_cls_token
        self.text_is_regressive = text_is_regressive

        if 'projection' not in kwargs:
            self.projection = 'default'
        else:
            self.projection = kwargs['projection']
        if self.projection == 'default':
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        elif self.projection == 'frozen_in_time':
            self.image_projection = nn.Sequential(
                nn.Linear(vision_width, embed_dim)
            )
            self.text_projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(text_width, embed_dim)
            )
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self.projection == 'default':
            nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection, std=self.text_width ** -0.5)
        else:
            nn.init.normal_(self.image_projection[0].weight, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection[1].weight, std=self.text_width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        if self.projection == 'default':
            x = x @ self.image_projection
        else:
            x = self.image_projection(x)

        return x

    def encode_text(self, text, attention_mask=None, use_checkpoint=False):
        if use_checkpoint:
            if isinstance(self.textual, DistilBertModel):
                pass
                # print("DistilBertModel does not support gradient checkpointing. Skipping even if use_checkpoint=True")
            else:
                self.textual.gradient_checkpointing_enable()
        else:
            self.textual.gradient_checkpointing_disable()
        # text, attention_mask = text.squeeze(1), attention_mask.squeeze(1)
        # ^ uncomment this only when doing local debugging (distributed=False)
        x = self.textual(text, attention_mask=attention_mask)

        if self.text_is_regressive:
            # gpt-style
            x = x.last_hidden_state
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        else:
            # bert-style
            if self.text_use_cls_token:
                x = x.last_hidden_state
                x = x[torch.arange(x.shape[0]), 0, :]
            else:
                x = x.pooler_output
        if self.projection == 'default':
            x = x @ self.text_projection
        else:
            x = self.text_projection(x)

        return x

    def forward(self, image, text, mask=None, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, attention_mask=mask, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}

def CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE(
    num_frames=16, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, adapter_dim = 64, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    # Add adapters after transformer layers within SpaceTimeTransformer
    adapter_layers = [Adapter(1024, adapter_dim) if i >= 20 else nn.Identity() for i in range(24)]
    vision_model.adapters = nn.ModuleList(adapter_layers)

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased', force_download=True
    )

    # Add adapters after each transformer layer within DistilBertModel
    # for layer in text_model.transformer.layer:
    #     layer.adapter = Adapter(768, adapter_dim)  # Add adapter to each transformer layer

    try:
        kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    except Exception:
        pass
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model