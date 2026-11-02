# =============================================================================
#  Anima Safe PAG — Forge Neo WebUI (Gradio 4) extension
# -----------------------------------------------------------------------------
#  Ported from the ComfyUI custom node "comfyui-anima-safe-pag".
#
#  Original repo: https://github.com/iljung1106/comfyui-anima-safe-pag
#
#  Port notes
#  ----------
#  * This Forge Neo backend has no "sampler_calc_cond_batch_function" hook
#    (backend/sampling/sampling_function.py calls calc_cond_uncond_batch
#    directly), so instead of padding the PAG row into one batch, the port
#    runs ONE extra forward pass inside sampler_post_cfg_function to obtain
#    the weak (perturbed) prediction. All guidance math is unchanged.
#  * Per-block attention patching is identical to the original node:
#    MethodType patching of SelfCrossAttention.compute_attention on
#    unet.model.diffusion_model.blocks[idx].self_attn
#  * UI follows the built-in "Sparse Attention Integrated" extension style:
#    an InputAccordion (the accordion itself is the enable switch, default
#    OFF) — no entry in the Settings page is needed, and toggling it takes
#    effect on the next generation without restarting the WebUI, because the
#    patch is applied to a per-pass clone of the UNet patcher (same pattern
#    as the built-in rescale_cfg / mahiro / sparse scripts).
# =============================================================================

from types import MethodType

import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

try:
    from gradio_rangeslider import RangeSlider
    _HAS_RANGE_SLIDER = True
except Exception:  # pragma: no cover - fallback: two separate sliders
    RangeSlider = None
    _HAS_RANGE_SLIDER = False

try:
    from backend.sampling.sampling_function import get_area_and_mult
except Exception:  # pragma: no cover - backend layout may differ
    get_area_and_mult = None


# ---------------------------------------------------------------------------
#  Presets.
#  "Default"  = values recommended by the original author in the node's README.
#  "Balanced" = softer variant: scale 1.3, active range 0.3 - 0.8, rest identical.
# ---------------------------------------------------------------------------
PRESET_DEFAULT = {
    "scale": 4.0,
    "block_indices": "18",
    "perturbation_strength": 0.75,
    "head_indices": "",
    "start_percent": 0.0,
    "end_percent": 0.7,
    "rescale": 0.20,
    "rescale_mode": "full",
}
PRESET_BALANCED = {
    **PRESET_DEFAULT,
    "scale": 1.3,
    "start_percent": 0.3,
    "end_percent": 0.8,
}
RECOMMENDED_PRESET = PRESET_DEFAULT  # backwards-compatible alias


# ===========================================================================
#  Algorithm helpers — ported 1:1 from comfyui-anima-safe-pag (__init__.py)
#  (original author iljung1106, MIT). Only the plumbing changed, not the math.
# ===========================================================================
def _sigma_to_float(sigma):
    if torch.is_tensor(sigma):
        return float(sigma.reshape(-1)[0].item())
    return float(sigma)


def _sigma_active(sigma, sigma_start, sigma_end):
    s = _sigma_to_float(sigma)
    a = _sigma_to_float(sigma_start)
    b = _sigma_to_float(sigma_end)
    if a < b:
        a, b = b, a
    return b <= s <= a


def _parse_indices(text, max_index):
    values = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = [p.strip() for p in part.split("-", 1)]
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise RuntimeError(f"Invalid block range '{part}'. Use values like 18 or 18,20,22.")
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    indices = sorted(i for i in values if 0 <= i <= max_index)
    if not indices:
        raise RuntimeError(f"No valid block indices found. Valid range is 0 to {max_index}.")
    return indices


def _parse_optional_indices(text, max_index):
    if text is None or not str(text).strip():
        return None
    return _parse_indices(text, max_index)


def _lerp_heads(base, target, strength, heads):
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return base
    if heads is None:
        return base.lerp(target, strength)
    out = base.clone()
    out[:, :, heads, :] = base[:, :, heads, :].lerp(target[:, :, heads, :], strength)
    return out


def _sdpa_attention(q, k, v):
    # Forge's SelfCrossAttention keeps tensors as [batch, seq, heads, dim]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    ).transpose(1, 2)


def _project_attention(attn_module, attn):
    value = attn.reshape(*attn.shape[:-2], attn.shape[-2] * attn.shape[-1])
    if hasattr(attn_module, "output_proj"):
        value = attn_module.output_proj(value)
        if hasattr(attn_module, "output_dropout"):
            value = attn_module.output_dropout(value)
        return value
    if hasattr(attn_module, "o_proj"):
        return attn_module.o_proj(value)
    raise RuntimeError("Unsupported attention module: no output projection found.")


def _make_pag_compute_attention(attn_module, original_compute, perturbation_strength, head_indices):
    """Soft-PAG replacement for SelfCrossAttention.compute_attention."""
    def compute_attention(self, q, k, v, transformer_options=None):
        if perturbation_strength <= 0.0:
            return original_compute(q, k, v, transformer_options=transformer_options or {})
        normal = _sdpa_attention(q, k, v)
        weak = _lerp_heads(normal, v, perturbation_strength, head_indices)
        return _project_attention(self, weak)

    return MethodType(compute_attention, attn_module)


def _patch_blocks(blocks, indices, perturbation_strength, head_indices):
    patched = []
    for idx in indices:
        block = blocks[idx]
        attn = getattr(block, "self_attn", None)
        if attn is None or not hasattr(attn, "compute_attention"):
            continue
        original = attn.compute_attention
        attn.compute_attention = _make_pag_compute_attention(
            attn, original, perturbation_strength, head_indices
        )
        patched.append((attn, original))
    return patched


def _restore_blocks(patched):
    for attn, original in patched:
        attn.compute_attention = original


def _rescale_guidance(guidance, cond_pred, cfg_result, rescale, mode):
    rescale = float(rescale)
    if rescale <= 0.0:
        return guidance
    guidance_result = cfg_result + guidance if mode == "full" else cond_pred + guidance
    reduce_dims = tuple(range(1, guidance_result.ndim))
    std_cond = torch.std(cond_pred, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    std_guidance = torch.std(guidance_result, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    factor = std_cond / std_guidance
    factor = rescale * factor + (1.0 - rescale)
    return guidance * factor


# ===========================================================================
#  Forge Neo plumbing
# ===========================================================================
def _pag_post_cfg(cfg_args, *, model, blocks, indices, heads, strength, scale,
                  sigma_start, sigma_end, rescale, rescale_mode, state):
    """
    Runs inside sampler_post_cfg_function (keys verified against
    sampling_function_inner's args dict: denoised/cond/uncond/model/
    cond_denoised/sigma/model_options/input). Performs one extra forward
    pass with soft-perturbed self-attention to get the weak prediction,
    then nudges the CFG result away from it.
    """
    denoised = cfg_args["denoised"]
    try:
        sigma = cfg_args["sigma"]
        if float(scale) <= 0.0 or not _sigma_active(sigma, sigma_start, sigma_end):
            return denoised

        cond_list = cfg_args.get("cond") or []
        model = model or cfg_args.get("model")
        if not cond_list or model is None:
            return denoised

        x = cfg_args["input"]
        timestep = cfg_args["sigma"]
        cond0 = cond_list[0]

        pag_input = x
        control = None
        c = None
        if get_area_and_mult is not None:
            processed = get_area_and_mult(cond0, x, timestep)
            if processed is not None:
                pag_input = processed.input_x
                control = processed.control
                c = {k: v.cond for k, v in processed.conditioning.items()}
        if not c:
            # fallback: process the raw model_conds to the batch size/device
            bs = x.shape[0]
            c = {}
            for k, v in cond0.get("model_conds", {}).items():
                try:
                    c[k] = v.process_cond(batch_size=bs, device=x.device).cond
                except Exception:
                    c[k] = getattr(v, "cond", v)

        # strip batch-synchronisation keys; keep patches etc. identical to the main pass
        mo = cfg_args.get("model_options", {}) or {}
        to = dict(mo.get("transformer_options", {}) or {})
        for k in ("cond_or_uncond", "sigmas", "cond_mark", "cond_indices",
                  "uncond_indices", "sampling_sigmas"):
            to.pop(k, None)

        patched = _patch_blocks(blocks, indices, strength, heads)
        try:
            pag_pred = model.apply_model(pag_input, timestep, control=control,
                                         transformer_options=to, **c)
        finally:
            _restore_blocks(patched)

        cond_pred = cfg_args.get("cond_denoised")
        if cond_pred is None or pag_pred.shape != cond_pred.shape:
            if not state.get("shape_warned"):
                print("[Anima Safe PAG] skipped one step: PAG prediction shape "
                      "does not match cond shape.")
                state["shape_warned"] = True
            return denoised

        guidance = (cond_pred - pag_pred) * float(scale)
        guidance = _rescale_guidance(guidance, cond_pred, denoised, rescale, rescale_mode)

        if not state.get("announced"):
            delta = (cond_pred - pag_pred).detach().abs().mean().item()
            print(
                "[Anima Safe PAG] active: "
                f"blocks={indices}, strength={float(strength):.3f}, "
                f"heads={'all' if heads is None else heads}, scale={float(scale):g}, "
                f"first-step mean |cond - pag| = {delta:.8f}"
            )
            state["announced"] = True

        return denoised + guidance
    except Exception as e:
        # never break sampling because of the PAG pass (e.g. OOM on low VRAM)
        print(f"[Anima Safe PAG] skipped this step due to: {e}")
        return denoised


class ScriptAnimaSafePAG(scripts.Script):
    """Renders like the built-in 'Sparse Attention Integrated' extension:
    a standalone InputAccordion in the txt2img/img2img column. Opening the
    accordion enables the patch (default: closed = OFF)."""

    def title(self):
        return "Anima Safe PAG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        with InputAccordion(False, label=self.title(), elem_id=f"{tab}_anima_safe_pag") as enable:
            # ---- Preset section (above PAG Scale) --------------------------
            gr.Markdown("**Preset**", elem_classes=["anima-safe-pag-section-title"])
            with gr.Row(elem_id=f"{tab}_anima_safe_pag_preset_row",
                        elem_classes=["anima-safe-pag-preset-row"]):
                default_btn = gr.Button(
                    "Default", variant="secondary", scale=1,
                    elem_classes=["anima-safe-pag-preset-btn"],
                    elem_id=f"{tab}_anima_safe_pag_preset_default",
                )
                balanced_btn = gr.Button(
                    "Balanced", variant="secondary", scale=1,
                    elem_classes=["anima-safe-pag-preset-btn"],
                    elem_id=f"{tab}_anima_safe_pag_preset_balanced",
                )

            # ---- Main controls ----------------------------------------------
            with gr.Row():
                scale = gr.Slider(
                    value=PRESET_DEFAULT["scale"], minimum=0.0, maximum=100.0, step=0.1,
                    label="PAG Scale", info="higher pushes the result farther from the weak prediction",
                    elem_id=f"{tab}_anima_safe_pag_scale",
                )
                perturbation = gr.Slider(
                    value=PRESET_DEFAULT["perturbation_strength"], minimum=0.0, maximum=1.0, step=0.001,
                    label="Perturbation Strength", info="0 = normal attention, 1 = hard PAG",
                    elem_id=f"{tab}_anima_safe_pag_perturb",
                )

            with gr.Row():
                block_indices = gr.Textbox(
                    value=PRESET_DEFAULT["block_indices"], lines=1, max_lines=1,
                    placeholder="18, 18-20, 18,20 ...",
                    label="Block Indices", info="blocks to perturb; > 14 recommended for 28-block Anima",
                    elem_id=f"{tab}_anima_safe_pag_blocks",
                )
                head_indices = gr.Textbox(
                    value=PRESET_DEFAULT["head_indices"], lines=1, max_lines=1,
                    placeholder="empty = all heads",
                    label="Head Indices", info="optional, e.g. 0-3, 8",
                    elem_id=f"{tab}_anima_safe_pag_heads",
                )

            if _HAS_RANGE_SLIDER:
                percent_range = RangeSlider(
                    minimum=0.0, maximum=1.0, step=0.001,
                    value=(PRESET_DEFAULT["start_percent"], PRESET_DEFAULT["end_percent"]),
                    label="Active Range", info="sampling progress: 0.0 = start, 1.0 = final step",
                    elem_id=f"{tab}_anima_safe_pag_range",
                )
                range_out = [percent_range]
            else:
                with gr.Row():
                    start_percent = gr.Slider(
                        value=PRESET_DEFAULT["start_percent"], minimum=0.0, maximum=1.0, step=0.001,
                        label="Start percent", elem_id=f"{tab}_anima_safe_pag_start",
                    )
                    end_percent = gr.Slider(
                        value=PRESET_DEFAULT["end_percent"], minimum=0.0, maximum=1.0, step=0.001,
                        label="End percent", elem_id=f"{tab}_anima_safe_pag_end",
                    )
                range_out = [start_percent, end_percent]

            with gr.Row():
                rescale = gr.Slider(
                    value=PRESET_DEFAULT["rescale"], minimum=0.0, maximum=1.0, step=0.01,
                    label="Rescale", info="0 = off; reduces contrast / saturation growth after PAG",
                    elem_id=f"{tab}_anima_safe_pag_rescale",
                )
                rescale_mode = gr.Dropdown(
                    ["full", "partial"], value=PRESET_DEFAULT["rescale_mode"],
                    label="Rescale Mode",
                    elem_id=f"{tab}_anima_safe_pag_mode",
                )

            # ---- Preset wiring ----------------------------------------------
            outputs = [scale, perturbation, block_indices, head_indices,
                       *range_out, rescale, rescale_mode]

            def _preset_updates(preset):
                updates = [
                    gr.update(value=preset["scale"]),
                    gr.update(value=preset["perturbation_strength"]),
                    gr.update(value=preset["block_indices"]),
                    gr.update(value=preset["head_indices"]),
                ]
                if _HAS_RANGE_SLIDER:
                    updates.append(gr.update(value=(preset["start_percent"], preset["end_percent"])))
                else:
                    updates.append(gr.update(value=preset["start_percent"]))
                    updates.append(gr.update(value=preset["end_percent"]))
                updates += [
                    gr.update(value=preset["rescale"]),
                    gr.update(value=preset["rescale_mode"]),
                ]
                return updates

            default_btn.click(fn=lambda: _preset_updates(PRESET_DEFAULT), inputs=[], outputs=outputs)
            balanced_btn.click(fn=lambda: _preset_updates(PRESET_BALANCED), inputs=[], outputs=outputs)

            # ---- infotext / API ----------------------------------------------
            self.infotext_fields = [PasteField(enable, "Anima Safe PAG", api="anima_safe_pag")]
            self.infotext_fields += [
                PasteField(scale, "Anima Safe PAG scale", api="anima_safe_pag_scale"),
                PasteField(perturbation, "Anima Safe PAG perturbation", api="anima_safe_pag_perturbation"),
                PasteField(block_indices, "Anima Safe PAG blocks", api="anima_safe_pag_blocks"),
                PasteField(head_indices, "Anima Safe PAG heads", api="anima_safe_pag_heads"),
            ]
            if _HAS_RANGE_SLIDER:
                self.infotext_fields.append(PasteField(percent_range, "Anima Safe PAG range", api="anima_safe_pag_range"))
            else:
                self.infotext_fields += [
                    PasteField(start_percent, "Anima Safe PAG start", api="anima_safe_pag_start_percent"),
                    PasteField(end_percent, "Anima Safe PAG end", api="anima_safe_pag_end_percent"),
                ]
            self.infotext_fields += [
                PasteField(rescale, "Anima Safe PAG rescale", api="anima_safe_pag_rescale"),
                PasteField(rescale_mode, "Anima Safe PAG rescale mode", api="anima_safe_pag_rescale_mode"),
            ]

            return [enable, scale, perturbation, block_indices, head_indices,
                    *range_out, rescale, rescale_mode]

    # ------------------------------------------------------------------
    # Runs before every sampling pass (txt2img / img2img / hires fix).
    # NOTE: called AFTER `forge_objects = forge_objects_after_applying_lora
    # .shallow_copy()` inside StableDiffusionProcessing*.sample(), so replacing
    # .unet here is per-pass only — closing the accordion disables the patch
    # on the very next generation, with no restart / model reload.
    # ------------------------------------------------------------------
    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) < 7:
            return
        enable = args[0]
        if not enable:  # accordion closed -> extension OFF (default)
            return

        scale = float(args[1])
        perturbation_strength = float(args[2])
        block_indices = args[3]
        head_indices = args[4]

        # args[5] is either a (start, end) tuple (RangeSlider) or the start float
        v5 = args[5]
        if isinstance(v5, (list, tuple)) and len(v5) == 2:
            start_percent, end_percent = float(v5[0]), float(v5[1])
            rescale, rescale_mode = float(args[6]), str(args[7])
        else:
            start_percent, end_percent = float(v5), float(args[6])
            rescale, rescale_mode = float(args[7]), str(args[8])

        if scale <= 0.0:
            return

        sd_model = p.sd_model
        forge_objects = getattr(sd_model, "forge_objects", None)
        if forge_objects is None:
            return
        unet = forge_objects.unet
        diffusion_model = getattr(unet.model, "diffusion_model", None)
        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks:  # only Anima-style DiT models expose .blocks
            return

        try:
            indices = _parse_indices(block_indices, len(blocks) - 1)
        except Exception as e:
            gr.Warning(f"Anima Safe PAG: {e}")
            return

        first_attn = getattr(blocks[indices[0]], "self_attn", None)
        n_heads = getattr(first_attn, "n_heads", 16)
        try:
            heads = _parse_optional_indices(head_indices, n_heads - 1)
        except Exception as e:
            gr.Warning(f"Anima Safe PAG: {e}")
            return

        predictor = getattr(unet.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "percent_to_sigma"):
            sigma_start = predictor.percent_to_sigma(max(0.0, min(1.0, start_percent)))
            sigma_end = predictor.percent_to_sigma(max(0.0, min(1.0, end_percent)))
        else:
            sigma_start = 1.0 - start_percent
            sigma_end = 1.0 - end_percent

        strength = max(0.0, min(1.0, perturbation_strength))
        state = {}

        def post_cfg(cfg_args):
            return _pag_post_cfg(
                cfg_args,
                model=cfg_args.get("model"),
                blocks=blocks,
                indices=indices,
                heads=heads,
                strength=strength,
                scale=scale,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                rescale=rescale,
                rescale_mode=rescale_mode,
                state=state,
            )

        # per-pass clone (same pattern as built-in rescale_cfg / mahiro / sparse scripts)
        new_unet = unet.clone()
        new_unet.set_model_sampler_post_cfg_function(post_cfg)
        p.sd_model.forge_objects.unet = new_unet

        p.extra_generation_params["Anima Safe PAG"] = (
            f"scale={scale:g}, blocks={block_indices}, strength={strength:g}, "
            f"heads={'all' if heads is None else head_indices}, "
            f"range={start_percent:.3f}-{end_percent:.3f}, "
            f"rescale={rescale:g} ({rescale_mode})"
        )
