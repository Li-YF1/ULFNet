
import torch
import torch.nn as nn
import dino.vision_transformer as vits
import torch.nn.functional as F
class DINOExtractor(nn.Module):
    def __init__(
            self,
            vit_model="dino",
            vit_arch="vit_base",
            vit_patch_size=8,
            enc_type_feats="k",
    ):
        super(DINOExtractor, self).__init__()
        self.vit_encoder, self.initial_dim, self.hook_features = get_vit_encoder(
            vit_arch, vit_model, vit_patch_size, enc_type_feats
        )
        self.vit_patch_size = vit_patch_size
        self.enc_type_feats = enc_type_feats
        if vit_arch=='vit_small':
            self.feat_dim=384
        else:
            self.feat_dim=768
        self.vit_feat='k'

        self.relu=nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(384, 197, kernel_size=1, stride=1,padding=0)
        self.bn1= nn.BatchNorm2d(197)
        self.conv2=nn.Conv2d(197,98,1,stride=1,padding=0)
        self.bn2=nn.BatchNorm2d(98)
        self.conv3=nn.Conv2d(98,48,1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(48)
        self.conv4=nn.Conv2d(48,1,1,stride=1,padding=0)

    def forward_step(self, batch,  for_eval=True):

        # Make the image divisible by the patch size
        if for_eval:
            batch = self.make_input_divisible(batch)
            _w, _h = batch.shape[-2:]
            _h, _w = _h // self.vit_patch_size, _w // self.vit_patch_size
        else:
            # Cropping used during training, could be changed to improve
            w, h = (
                batch.shape[-2] - batch.shape[-2] % self.vit_patch_size,
                batch.shape[-1] - batch.shape[-1] % self.vit_patch_size,
            )
            batch = batch[:, :, :w, :h]
        feat_h = batch.shape[-2] // self.vit_patch_size
        feat_w = batch.shape[-1] // self.vit_patch_size

        with torch.no_grad():

            attentions = self.vit_encoder.get_last_selfattention(batch)
            bs, nb_head, nb_token = attentions.shape[0], attentions.shape[1], attentions.shape[2]
            qkv = (
                self.hook_features["qkv"]
                .reshape(bs, nb_token, 3, nb_head, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            k = k.transpose(1, 2).reshape(bs, nb_token, -1)
            q = q.transpose(1, 2).reshape(bs, nb_token, -1)
            v = v.transpose(1, 2).reshape(bs, nb_token, -1)

            # Modality selection
            if self.vit_feat == "k":
                feats = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            elif self.vit_feat == "q":
                feats = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            elif self.vit_feat == "v":
                feats = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            elif self.vit_feat == "kqv":
                k = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
                q = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
                v = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
                feats = torch.cat([k, q, v], dim=1)

        feats = F.interpolate(feats, size=(feat_h * self.vit_patch_size, feat_w * self.vit_patch_size))
        return  feats

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        # From selfmask
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x


def get_vit_encoder(vit_arch, vit_model, vit_patch_size, enc_type_feats):
    if vit_arch == "vit_small" and vit_patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_small" and vit_patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_base" and vit_patch_size == 16:
        if vit_model == "clip":
            url = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        elif vit_model == "dino":
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        initial_dim = 768
    elif vit_arch == "vit_base" and vit_patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        initial_dim = 768

    if vit_model == "dino":
        vit_encoder = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0)
        # TODO change if want to have last layer not unfrozen
        for p in vit_encoder.parameters():
            p.requires_grad = False
        vit_encoder.eval().cuda()  # mode eval
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        vit_encoder.load_state_dict(state_dict, strict=True)

        hook_features = {}
        if enc_type_feats in ["k", "q", "v", "qkv", "mlp"]:
            # Define the hook
            def hook_fn_forward_qkv(module, input, output):
                hook_features["qkv"] = output

            vit_encoder._modules["blocks"][-1]._modules["attn"]._modules[
                "qkv"
            ].register_forward_hook(hook_fn_forward_qkv)
    else:
        raise ValueError("Not implemented.")

    return vit_encoder, initial_dim, hook_features





