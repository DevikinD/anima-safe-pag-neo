# Anima Safe PAG

Port of **[comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag)**
for [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) neo branch only.
* ported by AI

## Manual Installation

```bash
cd sd-webui-forge-classic/extensions
git clone https://github.com/DevikinD/anima-safe-pag-neo.git
```

## Usage
The extension renders as a standalone **"Anima Safe PAG"**
accordion in the txt2img / img2img column, right next to "Sparse Attention
Integrated" and the other `*_Integrated` extensions (same `InputAccordion`
pattern from `modules/ui_components.py`).

- **The accordion itself is the on/off switch** — closed = OFF (default).
- Toggling it takes effect on the **next generation, no restart**: the patch
  is applied to a per-pass clone of the UNet patcher inside
  `process_before_every_sampling`, so the base model objects are never
  modified.
- No entry is added to the Settings page.

## Presets
Two one-click presets sit in the **Preset** section at the top of the accordion:

| Preset | PAG Scale | Active Range | rest |
|---|---|---|---|
| **Default** | 4.0 | 0.0 – 0.7 | blocks `18`, strength `0.75`, all heads, rescale `0.20` full |
| **Balanced** | 1.3 | 0.3 – 0.8 | same as Default |

For 28-block Anima checkpoints, block indices above `14` are recommended
(`16`, `18`, `20`, `18-20` are good nearby tests).

## Layout / mobile
Controls are grouped in `gr.Row`s which stack vertically on narrow screens;
the active range uses a `RangeSlider` (two sliders fallback if
`gradio_rangeslider` is unavailable). All parameters are written to the image
infotext and exposed via the API (`anima_safe_pag_*`).

## Implementation note
This Forge Neo backend has no `sampler_calc_cond_batch_function` hook (unlike
ComfyUI), so the port computes the weak prediction with **one extra forward
pass inside `sampler_post_cfg_function`** instead of padding the cond batch.
The perturbation math (`scaled_dot_product_attention` → per-head lerp toward
`v` → output projection) and the guidance/rescale math are ported 1:1.
The extra pass is wrapped in try/except, so it can never abort sampling
(e.g. OOM on low VRAM). Only Anima-style models (`diffusion_model.blocks`)
are affected; other models are skipped automatically.
