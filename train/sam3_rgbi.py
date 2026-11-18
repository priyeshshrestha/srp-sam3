"""SAM 3 fine-tuned for FLAIR semantic segmentation, with RGB or RGB+NIR input.

The question this answers: does SAM 3 get better when it is TRAINED to use the NIR band? The answer
is the difference between two runs that differ ONLY in the input (`in_channels` 3 vs 4): same
trainable parts, same data order, same augmentations, same seeds.

What changes in SAM 3
  * Input: the 14x14 patch-embedding conv is widened from 3 to 4 channels. RGB filters are copied,
    NIR filters start at ZERO, so at step 0 the RGBI model computes exactly what pretrained SAM 3
    computes (verified on facebook/sam3: max |diff| = 0.0). The NIR filters still receive gradient.
  * Trainable: the patch embedding (both arms), LoRA adapters on every ViT attention projection
    (B starts at zero -> also an exact no-op at step 0), the DETR encoder (where the class phrase
    meets the image) and the semantic path of the mask decoder. Everything else is frozen.
  * Output: SAM 3's built-in semantic head (`semantic_seg`, one map per prompt). FLAIR classes are
    'stuff', so a dense map fits better than instance masks (frozen SAM 3 returned nothing on 34%
    of few-shot episodes). Each class is one fixed text prompt (models/sam3_fss.CLASS_PHRASES);
    at test time all 12 prompts are run and the per-pixel argmax is the prediction.
  * The text encoder (353 M) is run once for the 12 phrases and then dropped from memory.

  * Optional NDVI adapter (arms `rgb_ndvi` and `rgbi_ndvi`, after VegSAM, Wu, Li & Hong, CVPRW 2026): NDVI = (NIR - R) /
    (NIR + R) is patch-embedded to the ViT's token grid, passed through a per-block MLP and a shared MLP
    (zero-initialised, so again an exact no-op at step 0) and ADDED to the output of every ViT block.
    rgb_ndvi (VegSAM-style): the ViT sees only R,G,B and NIR enters ONLY through the adapter. rgbi_ndvi keeps
    the raw NIR channel as well, so rgbi_ndvi vs rgbi isolates what the NDVI prior adds on top of raw NIR.

`TinySeg` has the same interface and exists ONLY so run_train.sh can be tested without weights.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam3_fss import CLASS_PHRASES, SAM3_MODEL_ID

CLASSES = list(range(1, 13))          # FLAIR #1 ids of the 12 evaluated classes; index k <-> class k + 1
SAM3_SIZE = 1008                      # SAM 3's native input side
PATCH_EMBED = "vision_encoder.backbone.embeddings.patch_embeddings.projection"


def normalise(x_uint8: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) uint8-valued -> SAM 3 normalisation ((x/255 - 0.5) / 0.5), applied to every
    channel including NIR (so a zero-initialised NIR filter is exactly a no-op)."""
    return (x_uint8.float() / 255.0 - 0.5) / 0.5


# --------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with B = 0 at init (identical output to the base layer)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scale


def add_lora(module: nn.Module, names=("q_proj", "k_proj", "v_proj", "o_proj"), rank=16, alpha=16.0) -> int:
    """Wrap every nn.Linear whose attribute name is in `names`. Returns how many were wrapped."""
    n = 0
    for parent in list(module.modules()):
        for attr, child in list(parent.named_children()):
            if attr in names and isinstance(child, nn.Linear):
                setattr(parent, attr, LoRALinear(child, rank, alpha))
                n += 1
    return n


def widen_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Copy of `conv` with extra input channels whose weights are zero."""
    if in_channels == conv.in_channels:
        return conv
    if in_channels < conv.in_channels:
        raise ValueError("can only widen")
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new.to(conv.weight.device, conv.weight.dtype)


def ndvi_from_normalised(images: torch.Tensor) -> torch.Tensor:
    """(B, >=4, H, W) SAM-normalised ([-1, 1]) R,G,B,NIR -> (B, 1, H, W) NDVI in [-1, 1]."""
    x = images.float() * 0.5 + 0.5
    red, nir = x[:, 0:1], x[:, 3:4]
    return (nir - red) / (nir + red + 1e-4)


class NdviAdapter(nn.Module):
    """VegSAM-style vegetation prior: P_i = MLP_shared(GELU(MLP_i(PatchEmbed(NDVI)))) for ViT block i.
    MLP_shared starts at zero, so the adapter is an exact no-op until trained."""

    def __init__(self, hidden: int, n_layers: int, bottleneck: int = 64, patch: int = 14):
        super().__init__()
        self.embed = nn.Conv2d(1, hidden, patch, patch)
        self.per_block = nn.ModuleList(nn.Linear(hidden, bottleneck) for _ in range(n_layers))
        self.shared = nn.Linear(bottleneck, hidden)
        self.act = nn.GELU()
        nn.init.zeros_(self.shared.weight)
        nn.init.zeros_(self.shared.bias)

    def tokens(self, ndvi: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) -> (B, H/14, W/14, hidden), the layout of the ViT's hidden states."""
        return self.embed(ndvi).permute(0, 2, 3, 1)

    def forward(self, i: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.shared(self.act(self.per_block[i](tokens)))


# --------------------------------------------------------------------------------- the models
class Sam3Semantic(nn.Module):
    """forward(images (B, C, H, W) normalised, class_idx (B, P) in 0..11) -> logits (B, P, H, W)."""

    def __init__(self, in_channels: int = 3, lora_rank: int = 16, train_parts: bool = True,
                 model_id: str = SAM3_MODEL_ID, seed: int = 0, ndvi: bool = False,
                 input_channels: int = None):
        super().__init__()
        from transformers import Sam3Model, Sam3Processor
        input_channels = in_channels if input_channels is None else input_channels
        if ndvi and input_channels < 4:
            raise ValueError("the NDVI adapter needs the NIR band (input_channels >= 4)")
        if input_channels < in_channels:
            raise ValueError("input_channels must be >= in_channels")
        model = Sam3Model.from_pretrained(model_id)
        proc = Sam3Processor.from_pretrained(model_id)

        # Text features for the 12 phrases, computed once; then the text encoder is dropped.
        phrases = [CLASS_PHRASES[c] for c in CLASSES]
        tok = proc(text=phrases, return_tensors="pt", padding="max_length")
        with torch.no_grad():
            feats = model.get_text_features(input_ids=tok["input_ids"],
                                            attention_mask=tok["attention_mask"]).pooler_output
        self.register_buffer("text_feats", feats.float(), persistent=False)          # (12, L, 256)
        self.register_buffer("text_mask", tok["attention_mask"].bool(), persistent=False)
        model.text_encoder = None
        self.phrases = phrases

        parent_name, attr = PATCH_EMBED.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, widen_conv(getattr(parent, attr), in_channels))
        model.config.vision_config.backbone_config.num_channels = in_channels

        for p in model.parameters():
            p.requires_grad_(False)
        self.n_lora = 0
        if train_parts:
            # Re-seed here: widening the conv consumes RNG for the 4-channel run only, and the LoRA
            # initialisation must be identical for the RGB and RGBI runs of one seed.
            torch.manual_seed(seed)
            self.n_lora = add_lora(model.vision_encoder.backbone, rank=lora_rank, alpha=float(lora_rank)) if lora_rank else 0
            for name, p in model.named_parameters():
                if (name.startswith(PATCH_EMBED) or "lora_" in name or name.startswith("detr_encoder.")
                        or (name.startswith("mask_decoder.")
                            and not name.startswith(("mask_decoder.instance_projection", "mask_decoder.mask_embedder")))):
                    p.requires_grad_(True)
        self.model = model
        self.in_channels = in_channels            # channels the ViT's patch embedding reads (the first ones)
        self.input_channels = input_channels      # bands in the input tensor (NDVI needs R and NIR)
        self.hidden = model.config.detr_encoder_config.hidden_size
        self.ndvi_adapter = None
        self._ndvi_tokens = None
        self._ckpt = False
        if ndvi:
            torch.manual_seed(seed + 1)               # adapter init is independent of everything above
            self.ndvi_adapter = NdviAdapter(model.config.vision_config.backbone_config.hidden_size,
                                            len(model.vision_encoder.backbone.layers))
        self._install_layer_wrappers()

    def _install_layer_wrappers(self):
        """ViT layer i computes   out = layer_i(x)   [recomputed in backward when checkpointing]
        and, with the NDVI adapter,   out = out + adapter(i, ndvi_tokens)   OUTSIDE the checkpoint."""
        from torch.utils.checkpoint import checkpoint
        for i, layer in enumerate(self.model.vision_encoder.backbone.layers):
            if not hasattr(layer, "_orig_forward"):
                layer._orig_forward = layer.forward

            def fwd(x, *args, _l=layer, _i=i, **kw):
                if self._ckpt and _l.training and torch.is_grad_enabled():
                    out = checkpoint(_l._orig_forward, x, *args, use_reentrant=False, **kw)
                else:
                    out = _l._orig_forward(x, *args, **kw)
                if self.ndvi_adapter is not None:
                    out = out + self.ndvi_adapter(_i, self._ndvi_tokens).to(out.dtype)
                return out
            layer.forward = fwd

    def enable_checkpointing(self):
        """Recompute each ViT and DETR-encoder layer in the backward pass instead of storing its
        activations. Sam3VisionModel refuses HF's gradient_checkpointing_enable(), so each layer's
        forward is wrapped directly (only while training). The DETR encoder attends over all
        72x72 = 5184 tokens per prompt; without checkpointing, a kernel that materialises the
        attention matrix stores ~0.4 GB per layer per prompt (out of memory on a 20 GB Mac)."""
        from torch.utils.checkpoint import checkpoint
        self._ckpt = True                              # ViT layers: handled by _install_layer_wrappers
        for layer in list(self.model.detr_encoder.layers):
            def wrapped(*args, _f=layer.forward, _layer=layer, **kw):
                if _layer.training and torch.is_grad_enabled():
                    return checkpoint(_f, *args, use_reentrant=False, **kw)
                return _f(*args, **kw)
            layer.forward = wrapped

    def forward(self, images: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images), class_idx)

    def encode(self, images: torch.Tensor):
        """The image part (ViT + FPN), computed once per image whatever the number of prompts."""
        if images.shape[-1] != SAM3_SIZE or images.shape[-2] != SAM3_SIZE:
            images = F.interpolate(images, size=(SAM3_SIZE, SAM3_SIZE), mode="bilinear", align_corners=False)
        if self.ndvi_adapter is not None:
            self._ndvi_tokens = self.ndvi_adapter.tokens(ndvi_from_normalised(images))
        vis = self.model.vision_encoder(images[:, :self.in_channels])
        return list(vis.fpn_hidden_states[:-1]), list(vis.fpn_position_encoding[:-1])

    def decode(self, enc_img, class_idx: torch.Tensor) -> torch.Tensor:
        """The prompt part: class_idx (B, P) -> logits (B, P, h, w).
        Same wiring as Sam3Model.forward (transformers 5.x), minus the DETR decoder / box heads,
        which the semantic head does not use (checked against model(...).semantic_seg: max diff 2.5e-5)."""
        m = self.model
        B, P = class_idx.shape
        fpn = [t.repeat_interleave(P, 0) for t in enc_img[0]]
        pos = [t.repeat_interleave(P, 0) for t in enc_img[1]]
        flat = class_idx.reshape(-1)
        text = self.text_feats[flat].to(fpn[-1].dtype)
        tmask = self.text_mask[flat]
        enc = m.detr_encoder(vision_features=[fpn[-1]], text_features=text,
                             vision_pos_embeds=[pos[-1]], text_mask=tmask)
        queries = text.new_zeros(B * P, 1, self.hidden)
        out = m.mask_decoder(decoder_queries=queries, backbone_features=list(fpn),
                             encoder_hidden_states=enc.last_hidden_state,
                             prompt_features=text, prompt_mask=tmask)
        seg = out.semantic_seg                                           # (B*P, 1, h, w)
        return seg.reshape(B, P, *seg.shape[-2:])


class TinySeg(nn.Module):
    """Pipeline stand-in (NOT a model to report): conv features + a learned embedding per class."""

    def __init__(self, in_channels: int = 3, ndvi: bool = False, input_channels: int = None, **_):
        super().__init__()
        self.in_channels = in_channels
        self.input_channels = in_channels if input_channels is None else input_channels
        self.ndvi_conv = None
        if ndvi:
            self.ndvi_conv = nn.Conv2d(1, 16, 3, padding=1)
            nn.init.zeros_(self.ndvi_conv.weight)
            nn.init.zeros_(self.ndvi_conv.bias)
        self.body = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.cls = nn.Embedding(len(CLASSES), 16)
        self.phrases = [CLASS_PHRASES[c] for c in CLASSES]
        self.n_lora = 0

    def enable_checkpointing(self):
        pass

    def forward(self, images, class_idx):
        return self.decode(self.encode(images), class_idx)

    def encode(self, images):
        f = self.body(F.avg_pool2d(images[:, :self.in_channels], 4))              # (B, 16, h, w)
        if self.ndvi_conv is not None:
            f = f + self.ndvi_conv(F.avg_pool2d(ndvi_from_normalised(images), 4))
        return f

    def decode(self, f, class_idx):
        return torch.einsum("bchw,bpc->bphw", f, self.cls(class_idx))


def build(model: str, in_channels: int, lora_rank: int = 16, seed: int = 0, ndvi: bool = False,
          input_channels: int = None) -> nn.Module:
    if model == "sam3":
        return Sam3Semantic(in_channels=in_channels, lora_rank=lora_rank, seed=seed, ndvi=ndvi,
                            input_channels=input_channels)
    if model == "tiny":
        torch.manual_seed(seed)
        return TinySeg(in_channels=in_channels, ndvi=ndvi, input_channels=input_channels)
    raise ValueError(f"unknown model {model!r} (sam3 | tiny)")


def trainable_state(net: nn.Module) -> Dict[str, torch.Tensor]:
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if k in names}


def load_trainable(net: nn.Module, state: Dict[str, torch.Tensor]):
    own = {n for n, p in net.named_parameters() if p.requires_grad}
    missing, unexpected = own - set(state), set(state) - own
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(unexpected)[:5]} (different in_channels / lora_rank?)")
    net.load_state_dict(state, strict=False)


def predict_classes(net: nn.Module, images: torch.Tensor, out_size, chunk: int = 4) -> torch.Tensor:
    """(B, C, H, W) normalised -> (B, H, W) FLAIR #1 class ids 1..12 (argmax over the 12 prompts).
    The image is encoded once; prompts go through the decoder `chunk` at a time (bounded memory)."""
    B = images.shape[0]
    enc = net.encode(images)
    parts = []
    for start in range(0, len(CLASSES), chunk):
        idx = torch.arange(start, min(start + chunk, len(CLASSES)), device=images.device).unsqueeze(0).expand(B, -1)
        parts.append(F.interpolate(net.decode(enc, idx).float(), size=out_size, mode="bilinear", align_corners=False))
    return torch.cat(parts, 1).argmax(1) + 1
