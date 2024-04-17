## [Cross-Camera-Multi-Target-Vehicle-Tracking-Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/33)

AI-Driven Future of Transportation: Cross-Camera Multi-Target Vehicle Tracking Competition – Model Development Session


<details><summary>Hardware Information</summary>

- CPU: AMD Ryzen 5 5600X 6-Core @ 12x 3.7GHz
- GPU: NVIDIA GeForce RTX 3060 Ti (8G)
- RAM: 48087MiB

</details>


<details><summary>Create Conda Environment</summary>

```bash
$ conda create -n botsort python=3.7 -y
$ conda activate botsort

# https://pytorch.org/get-started/locally/

$ git clone https://github.com/ricky-696/AICUP_Baseline_BoT-SORT.git
$ cd AICUP_Baseline_BoT-SORT/
$ pip install numpy
$ pip install -r requirements.txt

# Install pycocotools
$ pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Cython-bbox
$ pip install cython_bbox

# faiss cpu / gpu
$ pip install faiss-cpu
$ pip install faiss-gpu
```

</details>


<details><summary>Folder Structure</summary>

```bash
MCMOT/
    ├── AICUP_Baseline_BoT-SORT/
    └── datasets/
        └── train/
```

</details>


<details><summary>Prepare ReID Dataset</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/

$ python fast_reid/datasets/generate_AICUP_patches.py --data_path ../datasets/train
# output: /home/wish/pro/AICUP/MCMOT/AICUP_Baseline_BoT-SORT/fast_reid/datasets/AICUP-ReID/
```

</details>


<details><summary>Prepare YOLOv7 Dataset</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/

$ python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir ../datasets/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
# output: /home/wish/pro/AICUP/MCMOT/AICUP_Baseline_BoT-SORT/datasets/AI_CUP_MCMOT_dataset/yolo
```

</details>


<details><summary>Download Pretrained Weight</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/
$ mkdir pretrained
$ cd pretrained/
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt
```

</details>


<details><summary>Train the ReID Module for AICUP</summary>

`fast_reid/configs/AICUP/bagtricks_R50-ibn.yml`
```
>> line 25: IMS_PER_BATCH: 60    # 256
```

```bash
$ cd AICUP_Baseline_BoT-SORT/

$ python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```

</details>


<details><summary>Fine-tune YOLOv7 for AICUP</summary>

- The dataset path is configured in `yolov7/data/AICUP.yaml`.
    ```
    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    train: /home/wish/pro/AICUP/MCMOT/AICUP_Baseline_BoT-SORT/datasets/AI_CUP_MCMOT_dataset/yolo/train
    val: /home/wish/pro/AICUP/MCMOT/AICUP_Baseline_BoT-SORT/datasets/AI_CUP_MCMOT_dataset/yolo/val
    
    # number of classes
    nc: 1
    
    # class names
    names: [ 'car' ]
    ```
- The model architecture can be configured in `yolov7/cfg/training/yolov7-AICUP.yaml`.
- Training hyperparameters are configured in `yolov7/data/hyp.scratch.custom.yaml` (default is yolov7/data/hyp.scratch.p5.yaml).


```bash
$ cd AICUP_Baseline_BoT-SORT/

# official
## finetune p5 models
$ python yolov7/train.py --device 0 --batch-size 16 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
## finetune p6 models
$ python yolov7/train_aux.py --device 0 --batch-size 16 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-w6-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-w6-AICUP --hyp data/hyp.scratch.custom.yaml

$ python yolov7/train.py --device 0 --batch-size 1 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e_training.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
```

</details>


<details><summary>Tracking and creating the submission file for AICUP (Demo)</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/

# Track one <timestamp> with BoT-SORT(-ReID) based YOLOv7 and multi-class (We only output class: 'car').
$ python3 tools/mc_demo_yolov7.py --weights runs/train/yolov7-AICUP/weights/best.pt --source /home/wish/pro/AICUP/MCMOT/datasets/train/images/0902_150000_151900 --device "0" --name "0902_150000_151900" --fuse-score --agnostic-nms --with-reid --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

$ Track all <timestamps> in the directory, you can execute the bash file we provided.
$ bash tools/track_all_timestamps.sh --weights runs/train/yolov7-AICUP/weights/best.pt --source-dir /home/wish/pro/AICUP/MCMOT/datasets/train/images --device "0" --fast-reid-config "fast_reid/configs/AICUP/bagtricks_R50-ibn.yml" --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
```

</details>


<details><summary>Evaluate your submission</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/

# Before evaluation, you need to run tools/datasets/AICUP_to_MOT15.py to convert ground truth into submission format:
$ python tools/datasets/AICUP_to_MOT15.py --AICUP_dir /home/wish/pro/AICUP/MCMOT/datasets --MOT15_dir /home/wish/pro/AICUP/MCMOT/datasets_MOT15

# You can use tools/evaluate.py to evaluate your submission by the following command:
$ cp 09*/*.txt tracking_results/
$ cp 10*/*.txt tracking_results/
$ python tools/evaluate.py --gt_dir /home/wish/pro/AICUP/MCMOT/datasets_MOT15 --ts_dir /home/wish/pro/AICUP/MCMOT/AICUP_Baseline_BoT-SORT/runs/detect/tracking_results
```

</details>


---


<details><summary>Fine-tune YOLOv9 for AICUP</summary>

```bash
$ cd AICUP_Baseline_BoT-SORT/

$ git clone https://github.com/WongKinYiu/yolov9.git
$ cd /yolov9
$ pip install seaborn thop
$ pip install ipython
$ pip install psutil

# demo
$ wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
$ python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c-converted.pt' --name yolov9_c_640_detect

# train
$ wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt
$ python train_dual.py --workers 8 --device 0 --batch 1 --data data/AICUP.yaml --img 1280 --cfg models/detect/yolov9-e.yaml --weights './yolov9-e-converted' --name yolov9-e --hyp hyp.scratch-high.yaml --min-items 0 --epochs 50 --close-mosaic 4
```

</details>


- Github Link for Baseline Model
    - https://github.com/ricky-696/AICUP_Baseline_BoT-SORT ([Legacy](https://github.com/ricky-696/AICup_MCMOT_Baseline))


### Acknowledgements

- [AICUP Baseline: BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT)
- [Official YOLOv9](https://github.com/WongKinYiu/yolov9)


### References

- [What is different in P5 model and P6 model? #141](https://github.com/WongKinYiu/yolov7/issues/141)
