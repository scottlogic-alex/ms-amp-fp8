# GPT3

We followed [`README.md`](README.md), and will note how we diverged from instructions.

> Install dependencies

We didn't install wikiextractor, because we will not use `prepare_wikipedia.sh` / the wikipedia dataset (it's big). We'll be using a Bulbapedia dataset from HF instead.

We install pybind11, because Megatron-DS compiles a dataset index builder (not convinced our HF dataset will actually _use_ this though).

```bash
pip install einops nltk pybind11
```

We now install apex, because DeepSpeed requires it.

## Installing apex

We had to install [apex](https://github.com/NVIDIA/apex).

```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

First, we installed its dependencies:

```bash
pip install -r requirements.txt
```

We had to [comment-out](https://github.com/NVIDIA/apex/pull/323) this [version check](https://github.com/NVIDIA/apex/blob/37d83fce4dcbb59897dfd951906493a6fe7fae37/setup.py#L40), because we had locally CUDA Toolkit 12.2, yet were using a PyTorch built against CUDA 12.1.

We then compiled:

```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```