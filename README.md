# Computation Graphs in PyTorch & JAX – Companion Repo

This repository accompanies the **“Dynamic vs. Static: Understanding Computation Graphs in PyTorch and JAX”** article on the Code Labs Academy blog.

It provides runnable, minimal code to explore exactly the ideas covered:

| Folder | What’s inside |
|--------|---------------|
| `examples/` | Pure‑PyTorch and pure‑JAX scripts that build, visualise, and debug small networks. |
| `utils/`    | Helper functions for dumping/autopsy of computation graphs in both frameworks. |
| `benchmarks/` | A micro‑benchmark that reproduces the latency table in the article. |

## Quick‑start


## Quick-Start

```bash
# 0. (optional but wise) upgrade pip so modern wheels resolve cleanly
python -m pip install --upgrade pip

# 1. Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate                     # Linux/macOS
# or: py -m venv .venv && .\.venv\Scripts\activate                    # Windows

# 2. Install dependencies (CPU-only default)
pip install -r requirements.txt

# 3. Run the examples
python examples/torch_example.py
python examples/jax_example.py
```

Apple-silicon note (M-series):
Make sure you’re running a native arm64 Python (e.g. Homebrew, Miniforge).
An x86/Rosetta interpreter will crash JAX with an “AVX instructions” error because all x86 wheels are compiled with AVX.
If you’re stuck on x86 Python, you can build jaxlib from source—but using a native arm64 Python is far easier and faster.

> **Tip:** If you’re on an Apple Silicon or CPU‑only machine, comment out the CUDA benchmark lines in `benchmark.py`.


**GPU optional:** The benchmark auto-detects CUDA.  
Force CPU with `DEVICE=cpu python benchmarks/benchmark.py`.

If you see giant “___kmpc_* undefined” linker warnings: `torch.compile` produces OpenMP kernels; on Apple M‑series chips you have to provide the runtime manually.

```bash
# 1. install LibOMP and LLVM via Homebrew
$ brew install libomp llvm

# 2. give Clang access to the runtime
#    a) simplest: symlink into LLVM's prefix
$ sudo ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib \
           /opt/homebrew/opt/llvm/lib/libomp.dylib

#    b) OR use a tiny wrapper that always links -lomp
$ mkdir -p ~/bin
$ cat <<'EOF' > ~/bin/clang++-omp
#!/usr/bin/env bash
exec /opt/homebrew/opt/llvm/bin/clang++ -fopenmp -lomp "$@"
EOF
$ chmod +x ~/bin/clang++-omp
$ export CXX=~/bin/clang++-omp

# 3. (optional) tell Torch‑Inductor where headers and libs live
$ export CPATH=/opt/homebrew/include
$ export LIBRARY_PATH=/opt/homebrew/lib
```

## Requirements

See `requirements.txt`. Tested with:

* Python 3.10
* PyTorch 2.3.0
* JAX 0.4.27 / jaxlib 0.4.27 + CUDA 12
* NVIDIA RTX 4090 (benchmarks) – but scripts also work on CPU.


## Running the Benchmark

```bash
# inside repo root, env activated
$ python benchmarks/benchmark.py

Batch    32 | **PyTorch:** 0.20 ms | **torch.compile:** 3.67 ms | **JAX:** 0.52 ms
Batch  2048 | **PyTorch:** 6.33 ms | **torch.compile:** 25.01 ms | **JAX:** 9.88 ms
```

## Learn More

* 📖 Read the accompanying blog post —  
  **[Dynamic vs. Static: Understanding Computation Graphs in PyTorch and JAX](https://codelabsacademy.com/blog/en/computation-graphs-pytorch-jax)**

* 🗂 Explore the [examples/](examples/) folder to see each graph in action.

* 🧪 New to automatic differentiation? The [FAQ](#faq) section links to papers and official docs.

* 🧑‍🎓 Bootcamp: [Data Science and AI Bootcamp](https://codelabsacademy.com/en/courses/data-science-and-ai/) – master ML essentials with live mentoring 

---

Made with ❤️ by Code Labs Academy.  
Feel free to open issues or PRs!
