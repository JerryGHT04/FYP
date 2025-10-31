# WarpX Installation Guide

This guide covers WarpX installation on both Linux PC and HPC systems. The main difference lies in dependency management.

> **📚 Official Documentation**: [WarpX CMake Installation Guide](https://warpx.readthedocs.io/en/latest/install/cmake.html#python-bindings-package-management)

## 📋 Prerequisites

### Required Dependencies
- **C++17 Compiler**: GCC 9.1+, Clang 7, NVCC 11.0, MSVC 19.15 or newer
- **CMake**: 3.24.0+
- **Git**: 2.18+
- **AMReX**: Automatically downloaded and compiled
- **PICSAR**: Automatically downloaded and compiled

### Python Binding Dependencies
- **pyAMReX**: Automatically downloaded and compiled
- **pybind11**: Automatically downloaded and compiled

## 🚀 Installation Steps

### Step 1: Navigate to Installation Directory
```bash
cd $HOME  # Or your preferred installation directory
```

### Step 2: Install Dependencies

<details>
<summary><b>🖥️ For Linux PC</b></summary>

Install dependencies using your package manager:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git python3 python3-venv python3-pip
```

</details>

<details>
<summary><b>🏢 For HPC Systems (Example: ICL HX3)</b></summary>

Load modules using the module manager. 

> ⚠️ **Important**: Ensure all modules use consistent GCC versions. For example, `Python/3.12.3-GCCcore-13.3.0` indicates Python 3.12.3 compiled with GCC 13.3.

**Required Modules:**
```bash
# Load these modules in order
module load GCCcore/13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load Python/3.12.3-GCCcore-13.3.0
```

**Save as Module Preset:**
```bash
# Save the loaded modules for future use
module save warpxGCC13_3

# Load saved preset in future sessions
module restore warpxGCC13_3
```

</details>

### Step 3: Download WarpX Source Code
```bash
git clone https://github.com/BLAST-WarpX/warpx.git src/warpx
cd src/warpx
```

### Step 4: Create Python Virtual Environment
```bash
# Create virtual environment (adjust path as needed)
python3 -m venv ../../warpx_2D_env

# Activate virtual environment
source ../../warpx_2D_env/bin/activate
```

> 💡 **Tip**: Create separate virtual environments for different build configurations (e.g., `warpx_2D_env`, `warpx_RZ_env`)

### Step 5: Install Python Dependencies
```bash
# Upgrade pip and essential build tools
python3 -m pip install -U pip
python3 -m pip install -U build packaging setuptools wheel
python3 -m pip install -U cmake

# Install WarpX requirements
python3 -m pip install -r requirements.txt
```

> ⚠️ **HPC Note**: Do NOT install cmake via pip on HPC systems (`python3 -m pip install -U cmake`). This would override the system-level cmake compiled with the correct GCC version.

### Step 6: Configure Build

Build only necessary features to speed up compilation. You can create multiple builds for different configurations.

#### Example: 2D Build with CUDA and Python Bindings
```bash
cmake -S . -B build_2D_py \
    -DWarpX_DIMS="2" \
    -DWarpX_PYTHON=ON \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_PARTICLE_PRECISION=SINGLE
```

> 💡 Full build options see official document

### Step 7: Build and Install Python Bindings
```bash
cmake --build build_2D_py --target pip_install -j 4
```

> 💡 **Performance Tip**: Adjust `-j N` where N is the number of CPU cores to use for compilation

## ✅ Verification

Test your installation:
```bash
# Activate virtual environment
source ../../warpx_2D_env/bin/activate

# Try import to verify installation
python3 -c "import pywarpx"
```

## 🔄 Managing Multiple Builds

You can maintain multiple builds for different use cases:
```bash
# 2D build with CUDA
cmake -S . -B build_2D_cuda -DWarpX_DIMS="2" -DWarpX_COMPUTE=CUDA

# RZ geometry build
cmake -S . -B build_RZ -DWarpX_DIMS="RZ" -DWarpX_PYTHON=ON
```

## 📝 Important Notes

1. **Always activate the correct Python environment** before running WarpX
2. **Create separate virtual environments** for different build configurations
3. **On HPC systems**, ensure module consistency across all dependencies
