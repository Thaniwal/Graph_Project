# GraphSteal Setup Complete

## Installation Summary

All dependencies have been successfully installed in the `GraphSteal` conda environment:

### Installed Packages
- ✅ All requirements from `requirements.txt` (with scipy 1.10.1 for Python 3.8 compatibility)
- ✅ RDKit (via conda)
- ✅ fcd_torch (for molecular metrics)
- ✅ Made graph-tool optional (not available on Windows via conda/pip)

### Code Modifications
- Made graph-tool imports optional (handles Windows compatibility)
- Fixed configuration: uncommented `test_only` in `general_default.yaml`

## Running the Code

### Method 1: Using PowerShell Script
From the project root:
```powershell
conda activate GraphSteal
$env:PYTHONPATH = $PWD
cd src
python run_train_diffusion.py
```

### Method 2: Step by Step

**Step 1: Pre-train graph diffusion model**
```powershell
conda activate GraphSteal
cd src
$env:PYTHONPATH = "C:\Users\haris\Desktop\GraphSteal"
python run_train_diffusion.py +experiment=debug  # For quick testing
# Or
python run_train_diffusion.py +experiment=qm9_with_h  # For full training
```

**Step 2: Train the classifier**
```powershell
python run_qm9_classifier.py
```

**Step 3: Reconstruct training graphs**
```powershell
python run_reconstruct.py
```

## Important Notes

1. **Dataset**: The QM9 dataset will be automatically downloaded to `./data` on first run
2. **Graph-tool**: Not required for basic functionality. Only used for certain graph statistics/metrics
3. **Experiments**: 
   - `debug` - Quick test run with small model
   - `qm9_with_h` - Full training with hydrogen atoms
   - `qm9_no_h` - Full training without hydrogen atoms
4. **Output**: Results will be saved in `../outputs/` directory (relative to src)

## Verification

The code has been verified to:
- ✅ Load successfully
- ✅ Download and process QM9 dataset (133,885 molecules)
- ✅ Initialize the model correctly

You can now proceed with training!


