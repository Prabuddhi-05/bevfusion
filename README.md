# BEVFusion Framework Setup and Execution

This repository provides setup instructions and usage guidelines for running the BEVFusion framework for **3D Object Detection using LiDAR and Multi-view Camera Fusion in the Bird's Eye View"** inside Docker. It is based on the [original BEVFusion repository by MIT HAN Lab](https://github.com/mit-han-lab/bevfusion).

---
## Docker container setup

Created and re-used a named Docker container (**bevfusion**) for convenient integration and consistent use inside Visual Studio Code.

---

## First-time user setup

### 1. Build the Docker image

Run this command only if the Dockerfile has been updated:

```bash
docker build -t bevfusion .
```

### 2. Create and run a named Docker container

```bash
docker run --gpus all -it \
  --name bevfusion-dev \
  -v "/media/prabuddhi/Crucial X9/bevfusion-main/data/nuscenes:/dataset" \
  --shm-size=16g bevfusion /bin/bash
```

* Replace the path with the actual dataset directory path on the host machine.

* Attach to this container via VS Code:

  * `Ctrl + Shift + P` → `Attach to an existing container`


### 3. Initial setup inside the container

Run the following commands:

```bash
cd /home
git clone https://github.com/mit-han-lab/bevfusion
OR
git clone https://github.com/Prabuddhi-05/bevfusion.git # Repo with all the modifications mentioned below (except numpy installation)
cd bevfusion

python setup.py develop # Install BEVFusion in development mode

mkdir -p data
ln -s /dataset ./data/nuscenes # Create a symbolic link to connect the dataset on the host machine to Docker 
```

### 4. Fix known issues

**Feature decorator issue**:

* Comment the following line in `/home/bevfusion/mmdet3d/ops/__init__.py`:

```python
# from .feature_decorator import feature_decorator
```

* In `/home/bevfusion/mmdet3d/models/backbones/radar_encoder.py`:

```python
# from mmdet3d.ops import feature_decorator
```

* In `/home/bevfusion/mmdet3d/models/backbones/__init__.py`:

```python
# from .radar_encoder import *
```

**NumPy attributeError**:

* Downgrade NumPy to resolve attribute errors:

```bash
conda install numpy=1.23.5 -y
```

### 5. Create swap memory (Prevents crashes)

* Check memory usage (optional):

```bash
htop
free -h
```

* Create a 64 GB swap file:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo bash -c "echo '/swapfile none swap sw 0 0' >> /etc/fstab"
```

### 6. Data preprocessing (Optional)

* Edit the preprocessing script to skip unnecessary steps:

* Comment out `create_groundtruth_database(...)` in `/home/bevfusion/tools/create_data.py`


### 7. Modify converter for all data types

* Modify `nuscenes_converter.py` in `/home/bevfusion/tools/data_converter` to preprocess all nuScenes data (LiDAR, Camera, and Radar data).

### 8. Run preprocessing:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### 9. Download pre-trained weights

```bash
./tools/download_pretrained.sh
```

### 10. Fix depth map channel mismatch

* Edit `/home/bevfusion/mmdet3d/models/vtransforms/depth_lss.py`:

Replace:

```python
d = self.dtransform(d)
```

With:

```python
if d.shape[1] != 1:
    d = d.mean(dim=1, keepdim=True)
d = self.dtransform(d)
```

### 11. Run evaluation

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox
```
### 12. Exit
```bash
exit

```

## Subsequent runs (Reuse container)

* Restart and reuse your named container without data loss:

```bash
docker start -ai bevfusion-dev
```

* You can directly re-run evaluations or training as required inside this container.

---

## Outputs

The model evaluates:

* **3D object detection** using fused **6-camera and LiDAR inputs**.
* Metrics include **NDS**, **mAP**, error metrics, and per-class results.

---

For detailed framework documentation, visit [BEVFusion GitHub](https://github.com/mit-han-lab/bevfusion).

