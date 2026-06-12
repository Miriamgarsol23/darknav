## Day 4 - Model implementation

### Architecture decision: LightEncoder vs ResNet-18

I implemented two encoders. LightEncoder is a custom 4-block CNN with 1.2M parameters
designed to be as lightweight as possible for CPU inference. ResNet-18 is the standard
choice from the computer vision literature with 11M parameters.

The key design constraint is that the decoder must be identical across all three
conditions so that the only variable is the encoder initialisation. This makes the
experiment clean: any difference in val IoU between conditions is attributable to the
pretraining, not to architecture differences.

I verified that both encoders produce the same 5 skip connection feature maps at the
same spatial resolutions, so the decoder UpBlocks connect identically.

### Single-channel adaptation of ResNet-18

ResNet-18 was designed for 3-channel RGB images. For DEM patches (single channel)
I adapted the first convolutional layer from (64, 3, 7, 7) to (64, 1, 7, 7).

For ImageNet pretraining (Condition B): initialise the new 1-channel conv by averaging
the pretrained RGB weights: w_new = w_rgb.mean(dim=1, keepdim=True).
This preserves the learned low-level feature detectors (edges, textures) while adapting
to grayscale input.

For NFW pretraining (Condition C): the encoder starts random, so no adaptation needed.

### Dimension verification

Ran verify_model_dimensions() for both modes. Output shape (1, 1, 128, 128) confirmed
for input (1, 1, 128, 128). The final_up + interpolate ensures spatial dimensions
match exactly regardless of rounding in the encoder pooling stages.

### Feature map visualisation

The feature maps from scratch initialisation show random noise with no structure,
as expected before training. After training I expect s0 (stem) to show edge detectors
and s3-s4 (deep layers) to show blob detectors sensitive to radially symmetric structures.
I will revisit this figure after training to compare scratch vs NFW pretrained features.

---

## Day 5 - Training

### Loss function: BCEWithLogitsLoss with pos_weight

Binary cross-entropy with logits is numerically more stable than BCE after sigmoid.
pos_weight computed in Day 3 (~9.0) upweights crater pixels by the class imbalance ratio.
Without this the model trivially learns all-background and achieves ~90% pixel accuracy
while completely failing at crater detection (IoU near 0).

### Learning rate strategy

Condition A (scratch): lr=1e-3, Adam, ReduceLROnPlateau patience=5
Condition B (imagenet): lr=1e-4 (10x lower), because pretrained weights should not
  be moved far from their ImageNet initialisation during the first epochs.
Condition C pretrain: lr=1e-3 on synthetic data
Condition C finetune: lr=1e-4 (same reasoning as B)

### Early stopping

patience=10 for main training, patience=15 for NFW pretraining (because the validation
metric on real data fluctuates more when training on synthetic domain).

### Observations after demo runs (15 epochs)

After 15 epochs the scratch model (A) shows clear learning: val IoU improves from
near 0 to a value that depends on how much real data is available. The imagenet model
(B) converges faster in the first 5 epochs due to the pretrained edge detectors.
The NFW pretrain phase shows that the model learns to detect radial structures on
synthetic data; whether this transfers to real data is the main research question
answered in Day 6.

### What the CLI interface enables

The train.py script can be run from the terminal for overnight full-epoch training
without Jupyter overhead. This is how I will run the full 50-epoch experiments:

  python src/train.py --condition A --epochs 50
  python src/train.py --condition B --epochs 50
  python src/train.py --condition C_pretrain --epochs 30
  python src/train.py --condition C_finetune --epochs 50 \
      --nfw_checkpoint runs/condition_C/nfw_pretrain_best.pth

History JSONs allow plotting training curves in the notebook independently of rerunning.
