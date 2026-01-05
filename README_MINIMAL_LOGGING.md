# Minimal Logging Training Scripts

## Overview

Three new training scripts have been created with minimal logging to reduce console clutter during training:

- `training_na_2_min.py` - NA (Node Action) mode with clean output
- `training_ea_2_min.py` - EA (Edge Action) mode with clean output
- `training_mh_min.py` - MH (Multi-Head) mode with clean output

## What Changed

### Suppressed Logging
- Environment initialization messages (billboard loading, graph creation, etc.)
- Model initialization messages (parameter counts, layer configurations)
- Tianshou internal updates ("Performing on-policy update...")
- PyTorch Geometric deprecation warnings
- Performance monitoring statistics

### What You Still See
- Training configuration summary at start
- Progress bar showing epoch/step progress
- New best reward notifications
- Training completion summary
- Errors and warnings (if they occur)

## Usage

### For Regular Training Runs
Use the minimal logging versions for clean output:

```bash
python training_na_2_min.py
python training_ea_2_min.py
python training_mh_min.py
```

### For Debugging
Use the original versions with verbose logging:

```bash
python training_na_2.py
python training_ea_2.py
python training_mh.py
```

### Command Line Arguments
All scripts support the same command line arguments:

```bash
python training_na_2_min.py --epochs 50 --lr 3e-4 --batch-size 64
python training_ea_2_min.py --epochs 100 --batch-size 128
python training_mh_min.py --epochs 120
```

## Example Output

### Before (Verbose - 100+ lines per epoch)
```
INFO:optimized_env:Initializing OptimizedBillboardEnv with action_mode=na
INFO:optimized_env:Loaded 444 billboard entries
INFO:optimized_env:Found 444 unique billboards
INFO:optimized_env:Loaded 5 advertiser templates
INFO:optimized_env:Processed trajectories for 1440 time points
INFO:optimized_env:Created graph with 108468 edges
[... repeated 6 times for parallel workers ...]
INFO:optimized_env:Environment reset with 4 initial ads
INFO:optimized_env:Environment reset with 3 initial ads
[... repeated 30+ times ...]
INFO:tianshou.trainer.base:Performing on-policy update on buffer of length 128
INFO:tianshou.trainer.base:Performing on-policy update on buffer of length 128
[... repeated 10-20 times per epoch ...]
```

### After (Minimal - 5-10 lines per epoch)
```
============================================================
Training Configuration:
  Mode: NA (Node Action)
  Billboards: 444, Max Ads: 20
  Epochs: 50, Steps/epoch: 10000
  Parallel envs: 4, Batch size: 64
  Device: cuda
============================================================
[Tianshou progress bar displays]
23:45:12 - INFO - New best reward: -2991.49, saving...
[Progress continues...]
23:47:35 - INFO - New best reward: -2850.12, saving...
[Progress continues...]
============================================================
Training complete! Best reward: -2450.33
Model saved to: models/ppo_billboard_na.pt
============================================================
```

## Temporary Verbose Logging in Minimal Version

If you need detailed logs temporarily while using the minimal version, add this at the top of the `main()` function:

```python
def main():
    # Temporarily enable verbose logging for debugging
    import logging
    logging.getLogger('optimized_env').setLevel(logging.INFO)
    logging.getLogger('models').setLevel(logging.INFO)
    logging.getLogger('tianshou.trainer.base').setLevel(logging.INFO)

    # ... rest of main function
```

## Notes

- TensorBoard logging is unaffected - all metrics are still captured
- Errors and warnings always appear regardless of logging level
- The progress bar from Tianshou continues to show detailed epoch/step information
- No performance impact - logging changes don't affect training speed
- Original scripts remain unchanged for reference and debugging
