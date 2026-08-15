# CRP-VLA Task 9 Provenance

- Repository revision before Task 9: `4c930306df2fb85c3ead0bcf4227eaa70fe443d0`
- Current repository revision before Task 9 commit: `4c930306df2fb85c3ead0bcf4227eaa70fe443d0`
- Branch: `task9-baseline-validation-preflight`
- Python: `3.12.13`
- Platform: `Linux-5.15.0-56-generic-x86_64-with-glibc2.35`
- GPU/runtime: `NVIDIA A100-PCIE-40GB, GPU-f0f74159-a195-9296-78e3-120bb1675174, 570.211.01, 40960 MiB`
- pip/uv freeze SHA-256: `58d048939779bef92fbf3ceefcdb9ccc747245c9da53ab983067924440a72286`
- Task 6 staged evidence origin revision: `ad689f200415201af145632cc8d1d146dac1f578`
- Confirmation primary rollout executed: `false`
- New training executed: `false`

## Verification commands

- `uv run pytest -q <Task 9 plus replay/manifest/hash regression tests>` → 35 passed
- `.venv/bin/ruff check <Task 9 scripts and test>` → passed
- `run_task9_analysis.py --audit-only` → INPUT_INTEGRITY_PASSED
- `run_task9_analysis.py --sample-trials 500` → TASK9_ANALYSIS_COMPLETED
- `build_task9_confirmation_manifest.py` attempt001 → admitted failure before state generation (byte-level processor manifest mismatch); attempt002 uses verified effective state/action contract
- `run_task9_preflight.py` attempt001 → admitted failure before policy inference (remote Hugging Face metadata request reset)
- `run_task9_preflight.py` attempt002 → admitted failure after model loading (incorrect LIBERO root supplied; no action or rollout executed)
- `run_task9_preflight.py` attempt003 → CONFIRMATION_PREFLIGHT_PASSED using the pinned LIBERO checkout and a Task 9-local offline VLM configuration

## Frozen dependency list

```text
absl-py==2.5.0
accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.14.2
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
ast-serialize==0.6.0
asttokens==3.0.2
async-lru==2.3.0
attrs==26.1.0
av==15.1.0
babel==2.18.0
bddl==1.0.1
beautifulsoup4==4.15.0
bleach==6.4.0
certifi==2026.7.22
cffi==2.1.0
cfgv==3.5.0
charset-normalizer==3.4.9
click==8.4.2
cloudpickle==3.1.2
cmake==4.1.3
comm==0.2.3
contourpy==1.3.3
coverage==7.15.3
cuda-bindings==12.9.7
cuda-pathfinder==1.6.0
cuda-toolkit==12.8.1
cycler==0.12.1
datasets==4.8.5
debugpy==1.8.21
defusedxml==0.7.1
dill==0.4.1
distlib==0.4.3
docopt==0.6.2
draccus==0.11.6
easydict==1.13
egl-probe==1.0.2
einops==0.8.2
etils==1.14.0
executing==2.2.1
farama-notifications==0.0.6
fastjsonschema==2.22.1
filelock==3.32.2
fonttools==4.63.0
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2026.2.0
future==1.0.0
gitdb==4.0.12
gitpython==3.1.57
glfw==2.10.2
grpcio==1.73.1
grpcio-tools==1.73.1
gymnasium==1.3.0
h11==0.16.0
h5py==3.16.0
hf-egl-probe==1.0.2
hf-libero==0.1.4
hf-xet==1.5.2
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.26.0
hydra-core==1.3.4
identify==2.6.19
idna==3.18
imageio==2.37.4
imageio-ffmpeg==0.6.0
iniconfig==2.3.0
ipykernel==6.31.0
ipython==9.16.0
ipython-pygments-lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
jedi==0.20.0
jinja2==3.1.6
json5==0.15.0
jsonlines==4.0.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-builder==1.2.1
jupyter-client==8.9.1
jupyter-console==6.6.3
jupyter-core==5.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter-server==2.20.0
jupyter-server-terminals==0.5.4
jupyterlab==4.6.2
jupyterlab-pygments==0.3.0
jupyterlab-server==2.28.0
jupyterlab-widgets==3.0.16
jupytext==1.19.5
kiwisolver==1.5.0
lark==1.3.1
-e file:///root/crp-vla
librt==0.13.0
llvmlite==0.48.0
markdown==3.10.3
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.11.1
matplotlib-inline==0.2.2
mdit-py-plugins==0.6.1
mdurl==0.1.2
mergedeep==1.3.4
mistune==3.3.4
mock-serial==0.0.1
mpmath==1.3.0
mujoco==3.8.1
multidict==6.7.1
multiprocess==0.70.19
mypy==2.3.0
mypy-extensions==1.1.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
nodeenv==1.10.0
notebook==7.6.1
notebook-shim==0.2.4
num2words==0.5.14
numba==0.66.0
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.19.0.56
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.1
opencv-python==4.14.0.94
opencv-python-headless==4.13.0.92
packaging==25.0
pandas==2.3.3
pandocfilters==1.5.1
parso==0.8.7
pathspec==1.1.1
pexpect==4.9.0
pillow==12.3.0
platformdirs==4.11.0
pluggy==1.6.0
pre-commit==4.6.1
prometheus-client==0.26.0
prompt-toolkit==3.0.53
propcache==0.5.2
protobuf==6.32.0
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.3
pyarrow==25.0.0
pycparser==3.0
pydantic==2.13.4
pydantic-core==2.46.4
pygments==2.20.0
pyopengl==3.1.10
pyparsing==3.3.2
pytest==9.1.1
pytest-cov==7.1.0
pytest-timeout==2.4.0
python-dateutil==2.9.0.post0
python-discovery==1.5.1
python-json-logger==4.1.0
pytz==2026.3.post1
pyyaml==6.0.3
pyzmq==27.1.0
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
robomimic==0.2.0
robosuite==1.4.0
rpds-py==2026.6.3
ruff==0.16.1
safetensors==0.8.0
scipy==1.18.0
send2trash==2.1.0
sentry-sdk==2.66.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
smmap==5.0.3
soupsieve==2.9.1
stack-data==0.6.3
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorboardx==2.6.5
termcolor==3.3.0
terminado==0.18.1
thop==0.1.1.post2209072238
tinycss2==1.5.1
tokenizers==0.22.2
toml==0.10.2
torch==2.11.0+cu128
torchcodec==0.11.1
torchvision==0.26.0+cu128
tornado==6.5.7
tqdm==4.70.0
traitlets==5.16.0
transformers==5.5.4
triton==3.6.0
typer==0.27.0
typing-extensions==4.16.0
typing-inspect==0.9.0
typing-inspection==0.4.2
tzdata==2026.3
uri-template==1.3.0
urllib3==2.7.0
virtualenv==21.7.1
wandb==0.27.2
wcwidth==0.8.2
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
werkzeug==3.1.8
widgetsnbextension==4.0.15
xxhash==3.8.1
yarl==1.24.5
zipp==4.1.0
```
