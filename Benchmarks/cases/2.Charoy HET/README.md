# 2D Hall Thruster PIC Benchmark

Replication of the benchmark case from:
> *"2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas"*  
> T. Charoy et al., 2019

Algorithm implementation follows:
> *"GPU-accelerated kinetic Hall thruster simulations in WarpX"*  
> Thomas A. Marks, 2025

## Requirements

Compile with 2D domain, CUDA, and single particle precision
```
cmake -S . -B build_2D_py \
    -DWarpX_DIMS="2" \
    -DWarpX_PYTHON=ON \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_PARTICLE_PRECISION=SINGLE
```


## Installation
```bash
# Navigate to this directory
cd '2.Charoy HET'

# Install Python dependencies
pip install -r requirements.txt
```

## Usage
```bash
python run_2D_Benchmark_final.py
```
On HPC, the job file is included. Run it with eg.
```
qsub run2DBenchmark_final.sh
```

## Output Files

### Preliminary Plots
- `Ex_plot.png` - Electric field (x-component)
- `ni_plot.png` - Ion density
- `phi_plot.png` - Electric potential
- `Te_plot.png` - Electron temperature

### Data Files
- Corresponding `.npy` files containing raw data points
- Use these for quantitative comparison with paper results

**Note:** Generated plots are preliminary visualizations only. For publication-quality figures, use the `.npy` data files to create custom plots matching the paper's format.

## Expected Runtime

~72 hours on NVIDIA A100 GPU

## Citation

If using this benchmark, please cite both papers listed above.