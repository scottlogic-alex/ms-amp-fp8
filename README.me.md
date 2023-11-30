Trying to do fp8 training on a 4090.

# Installing MS-AMP

First we installed MS-AMP from source, according to [these instructions](https://azure.github.io/MS-AMP/docs/getting-started/installation#install-from-source).  
We copy the commands here, so you can see anywhere where we have to diverge from or add to the instructions.

> Then, you can clone the source from GitHub.

```bash
git clone https://github.com/Azure/MS-AMP.git
cd MS-AMP
git submodule update --init --recursive
```

> If you want to train model with multiple GPU, you need to install MSCCL to support FP8. Please note that the compilation of MSCCL may take ~40 minutes on A100 nodes and ~7 minutes on H100 node.

We actually don't need multi-GPU support (4090 isn't intended to be used in a distributed setting), but we _do_ want FP8. It's unclear to us whether MSCCL is required for _single-GPU_ FP8, but we like compiling stuff so built it anyway.


```bash
cd third_party/msccl

# 4090
make -j src.build NVCC_GENCODE="-gencode=arch=compute_89,code=sm_89"

sudo apt-get update
sudo apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
sudo dpkg -i build/pkg/deb/libnccl2_*.deb
sudo dpkg -i build/pkg/deb/libnccl-dev_2*.deb
cd -
```

We make a new conda environment:

```bash
conda create -n llm-fp8 python=3.11
conda activate llm-fp8
```

> Then, you can install MS-AMP from source.

```bash
python3 -m pip install --upgrade pip

# Extra step we added..
# The below `pip install .` will try to compile mpi4py from source.
# Compiling mpi4py from source requires python and mpi headers, and mpi runtime libraries:
sudo apt install python-dev libopenmpi-dev
# Even with these headers, our attempts to compile mpi4py from source failed (linker failed to find mpi libraries)
# so we resorted to using the binary wheel they distribute
python -m pip install -i https://pypi.anaconda.org/mpi4py/simple mpi4py

# added MAX_JOBS var because it tries to install flash-attn. think it tries to build the wheel from source.
# it's an older version of flash-attn, so may predate the "crazy RAM requirements"/MAX_JOBS wisdom.
# likewise we specify where the CUDA compiler is, since some of the build-from-source dependencies didn't find this by default
# (again, may have been a problem limited to build-from-source of older versions of flash-attn)
CUDACXX=/usr/local/cuda/bin/nvcc MAX_JOBS=2 python3 -m pip install .
```

# Training GPT

See [`gpt/README.me.md`](gpt/README.me.md)