## [Cross-Camera-Multi-Target-Vehicle-Tracking-Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/33)

AI-Driven Future of Transportation: Cross-Camera Multi-Target Vehicle Tracking Competition – Model Development Session


<details><summary>Hardware Information</summary>

CPU: AMD Ryzen 5 5600X 6-Core @ 12x 3.7GHz
GPU: NVIDIA GeForce RTX 3060 Ti (8G)
RAM: 48087MiB

</details>


<details><summary>Create Conda Environment</summary>

```bash
$ conda create -n botsort python=3.7 -y
$ conda activate botsort

# https://pytorch.org/get-started/locally/

# Install pycocotools
$ pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Cython-bbox
$ pip3 install cython_bbox

# faiss cpu / gpu
$ pip3 install faiss-cpu
$ pip3 install faiss-gpu
```

</details>


- Github Link for Baseline Model
    - https://github.com/ricky-696/AICup_MCMOT_Baseline
    - https://github.com/ricky-696/AICUP_Baseline_BoT-SORT
