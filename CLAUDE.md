# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A knowledge-distillation research pipeline for autonomous-driving VLMs: a LoRA
fine-tuned **Qwen2.5-VL-7B-Instruct** "teacher" is distilled into a
**Qwen2.5-VL-3B-Instruct** "student" using two feature-level losses —
**L_spatial** (vision-encoder patch-level feature alignment) and **L_temporal**
(teacher-only multi-frame context distilled into a single-frame student) — on
top of the student's own task cross-entropy loss. Two ablation variants
(uniform-weight output KD, and no-distillation baseline) are trained alongside
the main method for comparison.

**Note on project history**: an earlier iteration of this pipeline used a
**hazard-weighted output-KD loss** (`HazardWeightedKDLoss`, weighting KL(student
‖ teacher) by an auto-generated scene-hazard label). That approach was replaced
by the current spatial/temporal feature-KD design — `train_distillation.py`
no longer contains `HazardWeightedKDLoss`. The hazard-labeling stage
(`data/hazard_labels.json`) and its label-generation script are still part of
the pipeline, but now serve a narrower, purely data-level role (see
`hazard_oversample` below) — they no longer feed any loss term.

## Setup

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0
pip install peft==0.14.0
pip install accelerate==1.4.0
pip install pillow
pip install qwen-vl-utils
```
Python 3.10. No requirements.txt/pyproject.toml exists — INSTALL.md is authoritative.

## Running the pipeline

Scripts are long-running (hours) and expected to be run via `nohup`, logged to
`logs/`, and monitored with a PID file — not run in the foreground. This is the
established pattern (RUNNING.md has worked examples, though its exact
`_2`-suffixed checkpoint/log names there predate the `_v2` naming below and
are stale — the `nohup ... & echo $! > ...pid` shape itself is still current)
for every stage:

```bash
nohup python scripts/<script>.py > logs/<name>.log 2>&1 & echo $! > logs/<name>.pid
tail -f logs/<name>.log
kill -0 $(cat logs/<name>.pid) && echo "실행 중" || echo "종료됨"   # check alive
kill $(cat logs/<name>.pid)                                        # stop
```

Pipeline order (each stage depends on the previous stage's checkpoint):

1. **`scripts/train_teacher.py`** — Stage 1: LoRA fine-tune the 7B teacher on
   DriveLM + NuScenes-QA. Produces `checkpoints/teacher_lora/epoch_N/`.
2. **`scripts/generate_hazard_labels.py`** — Stage 2: run the Stage-1 teacher
   over every DriveLM keyframe, prompting it to rate scene hazard 1–5. Produces
   `data/hazard_labels.json` (`{frame_token: score}`).
3. **`scripts/train_distillation.py`** — Stage 3 ("Full"/"Ours"): distill
   teacher → student 3B using `L_total = L_task + λ_spatial·L_spatial +
   λ_temporal·L_temporal` (no output-level KD term at all). Takes a
   `--variant {full,spatial,temporal}` CLI flag (default `full`) selecting a
   `VARIANTS` dict entry that overrides only `output_dir`/`lambda_spatial`/
   `lambda_temporal`/`temporal_k` on top of `CONFIG` — the rest of the script
   (data, LoRA, training loop) is identical across variants. This is
   deliberately a flag, not a duplicated script file (unlike `_v2` below):
   with only 4 values differing, physical duplication risks the same
   ablation-fairness bug class as the pre-`_v2` `lora_target_vision_attn` gap
   (one copy gets edited, the other forgotten). `full`
   (`temporal_k=2`, `spatial_layer_idx=23`, `λ_spatial=λ_temporal=1.0`) →
   `checkpoints/student_full/`; `spatial` (`λ_temporal=0`, and `temporal_k=1`
   since L_temporal=0 makes the teacher multi-frame input pointless — dropping
   to `temporal_k=1` also skips building `teacher_*` fields entirely, see
   dataloader notes below) → `checkpoints/student_spatial/`; `temporal`
   (`λ_spatial=0`, `temporal_k=2` unchanged since L_temporal still needs the
   teacher's multi-frame context) → `checkpoints/student_temporal/`. See
   "Loss functions" under Architecture below.
4. **`scripts/train_kd_only.py`** — ablation: identical distillation setup but
   with `UniformKDLoss` (every sample weighted equally, no output-KD in
   `train_distillation.py` anymore so this is now the only script that uses
   it). Produces `checkpoints/student_kd_only/` (v1), `checkpoints/student_kd_only_v2/`
   (v2 — vision-attn-LoRA fairness fix, see below), and `_v3`/`_v4` (KD
   hyperparameter fix — see "KD-only EOS collapse" note further below).
   **`student_kd_only_v4` is the checkpoint used in the 5-way comparison**;
   v2/v3 were a failure trail, not valid results, and their weights were
   **deleted on 2026-09-14 to free disk** — the diagnosis they supported is
   fully written up below and in `eval_results/`, so only the narrative remains.
5. **`scripts/train_baseline.py`** — ablation: student 3B fine-tuned directly
   on task loss only, no teacher involved. Produces
   `checkpoints/student_baseline/` (v1) and `checkpoints/student_baseline_v2/`
   (v2).

Steps 3–5 all require Stage 1's teacher checkpoint and Stage 2's hazard
labels. They can be run independently of each other once 1–2 are done.

**v1 vs v2 for the two ablation baselines**: v1 (`student_baseline`,
`student_kd_only`, both already trained) predates `train_distillation.py`'s
`lora_target_vision_attn` addition, so their LoRA `target_modules` only cover
the LLM decoder (`q/k/v/o_proj`, `gate/up/down_proj`) — a different trainable-
module set than `train_distillation.py`, which breaks strict ablation
fairness for the 5-way comparison. v2 adds the identical
`lora_target_vision_attn=True` flag to `train_baseline.py`/`train_kd_only.py`
(same flag, same default target-module extension as `train_distillation.py`;
everything else — batch size, LR, warmup, LoRA rank/alpha, sampler config —
stays frozen so exactly one variable changes) and writes to the `_v2`
output dirs so v1 is never overwritten. **v1 is intentionally kept**, not
because it's still used in the main 5-way table, but because v1-vs-v2 is
itself a free control experiment isolating "vision-attention LoRA capacity
alone" from "the spatial/temporal loss terms," reported as an appendix.
Verify the flag took effect via the printed `print_trainable_parameters()`
count at startup: 37,152,768 (v1, LLM-only) vs 41,124,224 (v2, matches
`train_distillation.py` exactly).

**KD-only EOS collapse (`student_kd_only_v2` → `_v3` → `_v4`)**: v2
(`lambda_kd=1.0`, `kd_temperature=4.0`) trained normally (task loss on par
with the other variants) but scored **1.78%** on NuScenes-QA val, vs ~50%
for the other four checkpoints. Directly inspecting per-step generation
probabilities showed the content answer was usually still correct with high
confidence (e.g. `"yes"` p=0.90–0.99) but EOS (`<|im_end|>`, id 151645) had
collapsed to a probability of ~1e-4–1e-6 (rank in the hundreds-of-thousands)
at every generation step, so the model never learned to stop and instead
appended runaway numeric/bounding-box-looking tokens after the correct
answer. **First hypothesis (lowering `lambda_kd` alone) was tried and
falsified**: v3 (`lambda_kd=0.3`, `kd_temperature=4.0` unchanged) fixed the
raw loss-magnitude ratio (`loss_kd`'s contribution dropped below
`loss_task`) but showed the *identical* EOS collapse at a checkpoint
(`step_5000`, 18.5% through training) where `student_baseline_v2` had
already learned clean EOS emission — proving the problem wasn't loss
magnitude. **Root cause, found by probing the teacher's own forced-position
logits**: at raw temperature (T=1) the teacher confidently predicts EOS at
the position right after the ground-truth answer (0.24–0.97 probability,
rank 1–2 across sample questions) — the teacher's target is fine. But
`UniformKDLoss` computes `softmax(logits / kd_temperature)` with
`kd_temperature=4.0` over the full ~152K-token vocabulary; re-probing at
T=4 showed the *same* teacher position's EOS probability crushed down to
~0.0004–0.0016 (still rank 1 but statistically indistinguishable from the
next few candidate tokens) — Hinton-style KD temperatures of 2–4 are
calibrated for ~1000-class problems (e.g. ImageNet), and applying the same T
to a vocabulary two orders of magnitude larger flattens the target into
near-uniform noise, destroying the "stop here" signal specifically (content
tokens, being comparatively less crowded in probability mass, survived the
flattening better). **Fix in `student_kd_only_v4`**: lower `kd_temperature`
to `1.5` (restoring target peakedness) and restore `lambda_kd` to the
standard `1.0` (no longer needed as a workaround once T was fixed; a lower T
also shrinks the loss's own `T²` scaling factor from 16× to 2.25×, so the
loss-magnitude balance improves as a side effect too). Verified via the same
forced-generation EOS probe at increasing checkpoints (`step_5000` through
`epoch_1`): EOS reaches rank 1 with 0.4–1.0 probability across sample
questions, matching baseline behavior. `checkpoints/student_kd_only_v4/`
is the checkpoint used in the final 5-way table (see Evaluation below).
`_v2` and `_v3` were the failure trail; **their weights were deleted on
2026-09-14** (disk pressure from ONNX exports) once the EOS-probe diagnosis
above was complete. Reproducing them would need a full retrain (~9.5 h each);
nothing downstream depends on them.

**Also deleted on 2026-09-14**: `checkpoints/student_distill/` — a pre-`_v2`
run whose `training_state.pt` still carried `lambda_hazard: 1.0`, i.e. the
retired `HazardWeightedKDLoss` design; it appears in no comparison.
`checkpoints_awq/` (16 GB) was **kept** — chapter 6's re-measurement with
`torch_dtype` and `max_pixels` held fixed has not been run yet, and the only
INT4 numbers on disk are the confounded ones. The v1 pair
(`student_baseline`, `student_kd_only`) was also **kept**: the v1-vs-v2
appendix control has no evaluation results yet.

The raw hazard-label distribution is heavily skewed (see
`logs/hazard_labels.log`: 88.6% of DriveLM keyframes score 2–3, only 0.3%
score 5), which used to mean rare high-hazard samples barely showed up during
training. `create_unified_dataloader()`'s `hazard_oversample` option (on by
default with `hazard_oversample_beta=0.5` in all training scripts) addresses
this by reweighting the `WeightedRandomSampler` *within* the DriveLM subset so
rarer scores are oversampled by roughly `(n_d/count)^beta`, normalized so the
overall `drivelm_ratio` split is unaffected. **This is purely a data-sampling
axis now** — no training script reads `batch["hazard_score"]` inside its loss
computation anymore (the old `HazardWeightedKDLoss` that did is gone). It's
kept on identically across `train_distillation.py`, `train_kd_only.py`, and
`train_baseline.py` only so all ablation variants see the same data
distribution and differ solely in loss function.

Checkpoints are saved as a standalone PEFT adapter (`save_pretrained`) plus the
processor and a `training_state.pt` with the config/loss used to produce it.
`train_distillation.py`, `train_kd_only.py`, and `train_baseline.py` save both
mid-epoch (`step_N/`, every `save_every_steps`) and at epoch end (`epoch_N/`),
and all three support resuming from a LoRA checkpoint via the `resume_from`
key in their `CONFIG` dict. **`train_teacher.py` is the exception**: it has no
`resume_from` key at all (no resume support) and only checkpoints at epoch end
(`epoch_N/`, cadence controlled by `save_every` in epoch units) — there is no
`step_N/` mid-epoch save for the teacher stage.

There is no test suite or linter configured. `scripts/check_*.py` are one-off,
gitignored data-inspection scripts (not part of any pipeline) — run directly,
e.g. `python scripts/check_labels.py`, to sanity-check dataset structure,
image-path resolution, or label masking after touching `dataloader.py`.

## Architecture

**`scripts/dataloader.py`** is the shared foundation every training script
imports from. It unifies two heterogeneous QA sources into one training
stream:

- `DriveLMDataset` reads the DriveLM-nuScenes JSON
  (`data/QA_dataset_nus/v1_0_train_nus.json`), which nests QA pairs as
  `scene → key_frames → frame → QA[task]` across four tasks (`perception`,
  `prediction`, `planning`, `behavior`). Each `(frame, task, qa_index)` becomes
  one sample.
- `NuScenesQADataset` reads `data/nuscenes_qa/NuScenes_train_questions.json`
  (flat question list keyed by `sample_token`), but has **no image data of its
  own** — it's joined to images via `build_token_to_images()`, which builds a
  `sample_token → {camera: abs_path}` map from the DriveLM JSON. Only
  NuScenes-QA samples whose token exists in that map are usable, so this
  dataset is implicitly a DriveLM-image subset.
- `create_unified_dataloader()` combines both via `ConcatDataset` +
  `WeightedRandomSampler`, targeting a fixed `drivelm_ratio` (default 0.4) of
  each batch regardless of the two datasets' very different sizes.
- Both datasets, when given a `processor`, tokenize question+answer as one
  sequence in a single `processor()` call (image included), then separately
  tokenize the question-only text (no image) purely to compute its token
  count — this two-pass trick gets `q_len` cheaply so the labels can be
  masked to `-100` up to the start of the answer, without a second full
  (image-inclusive) forward pass. `use_camera` defaults to `CAM_FRONT` only —
  the other 5 cameras' paths are loaded but unused.
- `hazard_score` flows through every sample (default `1.0` before Stage 2 has
  run) and rides along in `collate_fn` alongside the tensor batch, feeding
  only `create_unified_dataloader()`'s `hazard_oversample` reweighting
  (data-level; see above). No loss function reads it anymore.
- `temporal_k` (default `1` = off) is `train_distillation.py`-specific: when
  `>1`, `DriveLMDataset`/`NuScenesQADataset` additionally build a **teacher-only
  multi-frame input** — `build_scene_frame_order()` sorts each scene's
  `key_frames` chronologically (the JSON's own dict order is **not**
  chronological; sorting uses the microsecond timestamp embedded in each
  frame's `CAM_FRONT` filename — DriveLM keyframes are sparse, averaging
  ~3.1s apart, not the ~0.5s of raw nuScenes samples) and `_build_teacher_inputs()`
  tokenizes the current frame plus its `temporal_k-1` preceding frames
  (repeating the earliest frame if a scene has too few) into `teacher_input_ids`/
  `teacher_pixel_values`/`teacher_image_grid_thw`/`teacher_q_len`. Student's own
  fields are untouched — it always sees only the current single frame. Teacher
  tokenization uses **no fixed padding**; `make_collate_fn(pad_token_id)` pads
  `teacher_input_ids`/`teacher_attention_mask` dynamically to each *batch's own*
  max length (right-padded) rather than a global constant — a fixed
  `max_length=1600` global pad was measured to run ~18% slower wall-clock than
  per-batch dynamic padding at `batch_size=2`, even though per-sample content is
  usually far shorter (mean ≈515 tokens, p99 ≈717, true dataset max ≈1546 at
  `temporal_k=2`). `teacher_max_length` in CONFIG is now only a truncation
  safety cap, not the padding target.

**Loss functions** (defined inline in `train_distillation.py`, not shared):

- **`SpatialFeatureKDLoss`** — hooks `visual.blocks[spatial_layer_idx]`
  (default layer 23 of 32) on both teacher and student vision encoders, which
  are architecturally identical (`depth=32`, `hidden_size=1280`) but
  *not* naturally feature-aligned — an early attempt without any projection
  left the loss essentially flat (0.437→0.438) over 450 steps while `L_task`/
  `L_temporal` moved substantially in the same window, diagnosed as a capacity
  problem (a rank-16 vision LoRA alone can't both adapt representations *and*
  align two independently-pretrained feature spaces). The fix: a learnable
  `nn.Linear(1280, 1280)` (`self.proj`) projects the student's patches before
  the cosine loss, mirroring `TemporalContextKDLoss`'s existing pattern —
  after adding it, the same setup showed a clear, accelerating loss decrease
  from step 1. Since the teacher's vision-block output mixes patches from all
  `temporal_k` frames, an offset computed from `teacher_image_grid_thw`
  (patch count = `t*h*w` per image, pre-merge) slices out just the *current*
  frame's patches before comparing to the student's (single-frame) patches.
  Both `proj` and the vision-attention LoRA (see `lora_target_vision_attn`
  below) are removed/frozen at inference — they only exist to make training
  converge.
- **`TemporalContextKDLoss`** — teacher sees `temporal_k` frames (current +
  past), student sees only the current frame ("asymmetric" distillation, so
  the student's own inference cost never grows). Compares each model's final
  decoder-layer hidden state at the position right before answer generation
  (`q_len - 1`, valid regardless of any padding to its right since `q_len` is
  a left-anchored absolute index) via a learnable `nn.Linear(2048, 3584)`
  projecting student→teacher hidden size, then cosine loss.
- Both losses save their projection's `state_dict` as `spatial_proj.pt` /
  `temporal_proj.pt` alongside the LoRA adapter at every checkpoint, and
  reload it on `resume_from`.
- `HazardWeightedKDLoss`/`UniformKDLoss` still exist as-is in
  `train_kd_only.py` (`UniformKDLoss`, KL(student‖teacher) at temperature T,
  no weighting — the ablation baseline for "does *any* output-KD help") but
  `HazardWeightedKDLoss` is gone from `train_distillation.py` entirely. Both
  remaining KL-based losses truncate to `min(student_vocab, teacher_vocab)`
  before computing KL because the 3B and 7B models have different vocab sizes
  (151936 vs 152064) — this truncation logic only matters for
  `train_kd_only.py` now.

**LoRA target_modules**: all training scripts target
`q/k/v/o_proj + gate/up/down_proj`. `train_distillation.py` (always) and
`train_baseline.py`/`train_kd_only.py` (only in their `_v2` CONFIG, opt-in via
`lora_target_vision_attn`, default `False` there so old behavior is preserved
if the flag is omitted) additionally target vision-encoder `qkv`/`proj` when
the flag is `True` — worth noting that `gate/up/down_proj` **already**
unintentionally match the vision tower's MLP sublayers (`Qwen2_5_VLMLP` reuses
the same submodule names as the LLM decoder), so vision-encoder LoRA was
partially active even before `qkv`/`proj` were added; only vision *attention*
was fully frozen before this change.

**Training loop shape** is identical across all four training scripts
(train_teacher/train_distillation/train_kd_only/train_baseline): build
processor + model(s) → LoRA-wrap via `peft` → build the unified dataloader →
AdamW + warmup/cosine `LambdaLR` → `accelerate.Accelerator` with `bf16` mixed
precision and gradient accumulation for the effective batch size → manual loop
(not `Trainer`) logging every `log_every` steps and checkpointing at epoch end
(plus every `save_every_steps` mid-epoch, except in `train_teacher.py` — see
above). Where scripts differ is only in which model(s)/loss-module(s) are
loaded, whether a teacher forward pass runs (`torch.no_grad()`), and which
loss(es) combine with `s_out.loss` (the student's own CE loss is always
included as `loss_task`, even during distillation).
`train_distillation.py`'s loop logs a periodic `[grad check]` line (every
`log_every` steps) showing vision-LoRA and `spatial_proj` gradient norms — a
direct regression check against the flat/dead-gradient failure mode described
above.

**Hang detection (`train_distillation.py` only)**: a run once stalled for
~14 hours — GPU showed real utilization/power draw the whole time (not a
plain deadlock) but zero step-log progress, root cause never identified
because this sandboxed environment blocks `ptrace` (so `py-spy` and similar
external stack-inspection tools don't work on a running process here). Two
defenses now run every micro-step: (1) `checkpoints/student_full/
batch_fingerprint.log` appends one line per micro-step (tensor shapes +
`frame_token`s) *before* that batch is processed, so if a hang recurs the
last line identifies the exact batch that triggered it; (2)
`faulthandler.dump_traceback_later(1200, exit=True)` is armed at training
start and re-armed after every micro-step completes — if 20 minutes pass with
no micro-step finishing, it dumps every thread's Python stack to
`checkpoints/student_full/watchdog_stackdump.log` and force-exits, bounding
any future hang to ~20 minutes of wasted GPU time instead of indefinite. Both
are cheap (one small file write / one syscall per step) and were verified not
to affect training dynamics or throughput.

Both teacher and student are always LoRA-adapted (never full fine-tuned);
`checkpoints/*/epoch_N/` and `step_N/` directories hold adapter weights only
(~hundreds of MB), loaded relative to the frozen base model named in each
script's `CONFIG`.

`checkpoints/`, `data/`, `external/`, and `logs/` are all gitignored
(large/generated artifacts) — expect them to exist locally but never assume
they're tracked in git history. `external/EM-VLM4AD/` additionally has its
**own nested `.git`** (it was `git clone`d in directly, not added as a
submodule) — even edits inside it that *aren't* covered by a gitignore rule
would never show up in this repo's `git status`/`git add`, since a nested
`.git` directory boundaries off from the parent repo. Any change made there
(e.g. `modules/multi_frame_model.py`) only exists on this machine's disk —
if the instance is destroyed, redo it from `CHECKPOINT_PROVENANCE.md`'s
source/date rather than assuming it survived a git push.

## Evaluation

The original plan to evaluate on the official DriveLM test set hit a hard
constraint worth knowing before touching evaluation code:

- **DriveLM does not publicly release ground-truth answers for held-out
  nuScenes val/test scenes.** `data/QA_dataset_nus/v1_1_val_nus_q_only.json`
  (149 scenes, 799 keyframes, downloaded from the official challenge's Google
  Drive mirror — HuggingFace's copy is gated) has every `"A"` field empty
  (`"q_only"` in the filename is literal). The only real scoring path was the
  official leaderboard (`AGC2024/driving-with-language-official` HF Space),
  which was tied to the **CVPR 2024** Autonomous Grand Challenge and closed
  2024-06-01 (no 2025+ successor track, and submission requires a `team` name
  from that competition's now-defunct Google registration form) — **not
  usable**.
- The **main quantitative evaluation axis is instead NuScenes-QA val**
  (`data/nuscenes_qa/NuScenes_val_questions.json`), which does ship ground-truth
  answers. Only the subset whose `sample_token` matches a DriveLM val keyframe
  is usable (images are only available for those 799 keyframes — same
  DriveLM-subset-filtering pattern as the train side), giving **11,309** QA
  samples. Because DriveLM val scenes have zero overlap with the 696 DriveLM
  train scenes actually used for training, this is a clean scene-level
  held-out set. `scripts/eval_nuscenesqa_val.py` runs any student checkpoint
  against it (exact-match accuracy, normalized via `scripts/eval_utils.py`'s
  `normalize_answer()`, broken down by `template_type`: exist/object/status/
  count/comparison):
  ```bash
  python scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_full/epoch_1
  ```
- **5-way ablation results (NuScenes-QA val, 11,309 samples, exact-match,
  `eval_results/*_nuscenesqa_val.json`)** — the main comparison this whole
  pipeline was built to produce:

  | Model | Overall | comparison | count | exist | object | status |
  |---|---|---|---|---|---|---|
  | **Full (Ours)** — `student_full` | **51.69%** | 64.72% | 7.46% | 80.17% | **43.87%** | 51.03% |
  | L_spatial only — `student_spatial` | 51.40% | 63.09% | **10.58%** | 81.38% | 37.86% | **52.79%** |
  | L_temporal only — `student_temporal` | 51.34% | 64.44% | 7.73% | **81.17%** | 39.89% | 52.24% |
  | Baseline (no KD) — `student_baseline_v2` | 50.30% | 64.49% | 7.96% | 80.26% | 36.67% | 51.15% |
  | KD-only (uniform output-KD) — `student_kd_only_v4` | 46.74% | 63.88% | 10.40% | 67.68% | 35.48% | 51.09% |

  Takeaways: **the overall-accuracy ordering among Full / spatial / temporal
  (51.69 / 51.40 / 51.34%) is NOT statistically supported** — every variant was
  trained once with no seed control, so a 0.35pp spread is indistinguishable from
  seed noise. Do not write "Full edges out the single-loss ablations"; write that
  the three are indistinguishable on overall accuracy.

  **Correction (2026-09-14)**: this passage used to say "only the larger, *repeatable*
  gaps carry weight" about Full's `object` lead. **"repeatable" was unfounded** — that
  measurement was never repeated either. It is the same single-seed, n=1 observation as
  the 0.35pp spread, just a wider margin. Seed-variance measurement was deliberately
  dropped (5 variants × 3 seeds ≈ 774 h ≈ 32 days; see `thesis_outline_20260910.md` §7),
  so no gap in this table gets a confidence interval. The correct phrasing for Full's
  `object` lead (43.87% vs 37–40%): the margin is far wider than the 0.35pp overall
  spread and its direction matches L_align's patch-alignment objective, **but it is
  still a single-seed observation and must not be written as a ranking claim**. Report
  it as a directional observation, not as evidence that Full is better. `student_baseline_v2` (no distillation at
  all) is a strong baseline (50.30%), so the feature-KD gain over pure task
  fine-tuning is real but modest (~1–1.4pp). KD-only is the clear worst
  performer, especially on `exist` (67.68% vs ~80% for every other variant)
  — even after the EOS-collapse fix above, output-level uniform KD
  underperforms both feature-level KD and the plain baseline on this task,
  which is the headline result supporting this project's feature-KD-over-
  output-KD thesis. `count` is uniformly weak (7–11%) across all five
  variants — exact numeric counting appears to be a hard limit at this model
  scale/data regime rather than something any one loss design fixes.
- **EM-VLM4AD reference comparison (kept separate from the 5-way table above,
  not a 6th row)** — `external/EM-VLM4AD`'s T5-Medium/T5-Base checkpoint,
  fine-tuned on the same NuScenes-QA train subset our students see
  (`scripts/finetune_em_vlm4ad_nuscenesqa.py --single_camera`, 3 epochs) and
  evaluated with `scripts/eval_em_vlm4ad_nuscenesqa.py --single_camera
  --max_length 16` — `--single_camera` feeds the same CAM_FRONT image into
  all 6 of the model's view slots (repeated, not distinct views) so its
  visual input matches our students' CAM_FRONT-only setup, and
  `--max_length 16` matches our eval's `max_new_tokens=16`; these two flags
  exist specifically to remove the two biggest unfairness sources found when
  auditing this comparison (EM-VLM4AD natively gets 6 distinct camera views
  and an effectively unbounded 512-token generation cap, both unlike our
  students). Result (`eval_results/em_vlm4ad_T5-Medium_nuscenesqa_ft_singlecam_finetuned_singlecam_nuscenesqa_val.json`):
  **54.08%** overall (6116/11309) — `count` 18.35%, `exist` 80.50%, `object`
  42.65%, `comparison` 67.87%, `status` 50.48%. This is *higher* than
  `student_full`'s 51.69%, including a clear lead on `count` (18.35% vs
  7–11% for every one of our 5 variants). Kept as a separate note rather than
  merged into the 5-way table because it isn't a controlled ablation variant
  of our pipeline — it's a different model family (T5-Base LM, CNN/ViT
  visual front-end) with its own separately-pretrained DriveLM stage this
  project didn't run, so a higher number here doesn't undercut the 5-way
  ablation's internal conclusions (those only compare our own checkpoints
  against each other under identical training conditions). If asked "why not
  just use EM-VLM4AD" — this project's contribution is the distillation
  *method* (compressing a general-purpose 7B VLM teacher into a 3B student
  while preserving spatial/temporal understanding), not narrow leaderboard
  accuracy on one QA format; EM-VLM4AD has no analogous "teacher" to
  compress, is architecturally capped at fixed 224×224/6-view CNN input
  (no native dynamic resolution, no general instruction-following), and its
  NuScenes-QA fine-tuning measurably pushes it toward terser answers,
  trading away the more explanatory driving-QA style DriveLM training gives
  it — a property that matters for safety-relevant explainability in ways a
  closed-set exact-match score doesn't capture.
- DriveLM val (with its empty answers) is kept only as an image source for
  qualitative side-by-side samples, not for scoring.
- CODA-LM is a second, **largely** (not fully) domain-disjoint quantitative axis with
  published ground truth, used for zero-shot generalization evaluation. **Correction
  (2026-09-11)**: this line previously said "fully domain-disjoint (non-nuScenes)",
  which is wrong. CODA-LM reuses CODA images verbatim (its README: "Images of CODA-LM
  train set come from CODA2022 val set, while images of CODA-LM val and test sets come
  from CODA2022 test set"), and CODA itself was mined from three datasets — KITTI (309
  scenes), **nuScenes (134)**, and ONCE (1,057). So roughly 9% of the underlying scenes
  are nuScenes. Whether those specific nuScenes scenes overlap the 696 DriveLM train
  scenes we train on has **not** been checked; until it is, describe CODA-LM as
  "largely domain-disjoint" rather than held-out by construction.
  A 143-unique-image subset (CODA-LM llava-format Mini, English) is downloaded to
  `data/codalm_mini/` via `scripts/fetch_codalm_images.py` — note the raw Mini parquet
  has 193 rows but only 143 distinct images (the `general` and `suggestion` tasks share
  the same 50 scenes; `regional` adds 93 more). Images are pre-resized to ~720p, so
  they yield ~1,175 vision tokens versus nuScenes' 1,836 at the same `max_pixels` —
  any cross-dataset activation comparison must control for token count, not resolution.
- **SOTA comparison** (EM-VLM4AD, MiniDrive, etc.) is two-tiered:
  MiniDrive's GitHub repo (`EMZucas/minidrive`) has no code or weights at all
  (citation only, can't be reproduced). EM-VLM4AD does ship code+checkpoints
  (cloned into `external/EM-VLM4AD/`, checkpoints under
  `external/EM-VLM4AD/multi_frame_results/{T5-Medium,T5-Large}/` — see
  `CHECKPOINT_PROVENANCE.md` there for source/date). `scripts/eval_em_vlm4ad_nuscenesqa.py`
  runs it zero-shot on NuScenes-QA val, but EM-VLM4AD was only ever trained on
  DriveLM, so it answers in DriveLM's verbose conversational style (e.g.
  `"Yes, there are traffic cones in the front right."` vs the expected `"yes"`)
  — confirmed via smoke test, this tanks exact-match scores for reasons
  unrelated to capability:
  ```bash
  python scripts/eval_em_vlm4ad_nuscenesqa.py --model-name T5-Medium --lm T5-Base
  ```
  `scripts/finetune_em_vlm4ad_nuscenesqa.py` (not yet
  run — needs a free GPU window) fine-tunes it on the *same* NuScenes-QA train
  subset our student sees, to get a same-data/same-eval-set comparison instead
  (then re-run `eval_em_vlm4ad_nuscenesqa.py --model-name T5-Medium_nuscenesqa_ft
  --lm T5-Base --eval_mode finetuned` against the fine-tuned checkpoint).
  Published DriveLM-benchmark numbers for these models are also only usable as
  an indirect citation comparison, not a direct one — those numbers are
  themselves typically from a self-defined split of DriveLM *train* (the
  official test answers were never public even for those papers), so citing
  them needs the same caveat.

## Language

Code comments, print statements, and log messages throughout this repo are in
Korean — match that convention when editing existing scripts.
