# BEVFusion Framework Setup and Execution

This repository provides setup instructions and usage guidelines for running the BEVFusion framework for **3D Object Detection using LiDAR and Multi-view Camera Fusion** inside Docker. It is based on the [original BEVFusion repository by MIT HAN Lab](https://github.com/mit-han-lab/bevfusion).

---
## Docker Container Setup

Created and re-used a named Docker container (**bevfusion**) for convenient integration and consistent use inside Visual Studio Code.

---

## 🚀 First-time User Setup

### 1. Build the Docker Image

Run this command only if the Dockerfile has been updated:

```bash
docker build -t bevfusion .
```

### 2. Create and Run a Named Docker Container

```bash
docker run --gpus all -it \
  --name bevfusion-dev \
  -v "/media/prabuddhi/Crucial X9/bevfusion-main/data/nuscenes:/dataset" \
  --shm-size=16g bevfusion /bin/bash
```

* Attach to this container via VS Code:

  * `Ctrl + Shift + P` → `Attach to an existing container`

### 3. Initial Setup Inside the Container

Run the following commands:

```bash
cd /home
git clone https://github.com/mit-han-lab/bevfusion
OR
git clone https://github.com/Prabuddhi-05/bevfusion.git (with all the modifications)
cd bevfusion

python setup.py develop

mkdir -p data
ln -s /dataset ./data/nuscenes
```

### 4. Fix Known Issues

**Feature Decorator Issue**:

* Comment the following line in `/home/bevfusion/mmdet3d/ops/__init__.py`:

```python
# from .feature_decorator import feature_decorator
```

**NumPy AttributeError**:

* Downgrade NumPy to resolve attribute errors:

```bash
conda install numpy=1.23.5 -y
```

### 5. Create Swap Memory (Prevents Crashes)

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

### 6. Data Preprocessing (Optional, once per dataset)

Edit the preprocessing script to skip unnecessary steps:

* Comment out `create_groundtruth_database(...)` in `/home/bevfusion/tools/create_data.py`

Run preprocessing:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### 7. Modify Converter for All Data Types

* Modify `nuscenes_converter.py` to process LiDAR, Camera, and Radar data.

### 8. Download Pre-trained Weights

```bash
./tools/download_pretrained.sh
```

### 9. Fix Depth Map Channel Mismatch

Edit `mmdet3d/models/vtransforms/depth_lss.py`:

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

### 10. Run Evaluation

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox
```

---

## 🔄 Subsequent Runs (Reuse Container)

Restart and reuse your named container without data loss:

```bash
docker start -ai bevfusion-dev
```

You can directly re-run evaluations or training as required inside this container.

---

## 📌 Outputs

The model evaluates:

* **3D object detection** using fused **6-camera and LiDAR inputs**.
* Metrics include **NDS**, **mAP**, error metrics, and per-class results.

---

## ✅ Important Checks (For Reference)

* Verify consistency of results.
* Visualize bounding boxes to confirm detections.
* Thoroughly review LSS and Fusion-related code implementations.

For detailed framework documentation, visit [BEVFusion GitHub](https://github.com/mit-han-lab/bevfusion).

---

This README provides streamlined instructions for efficient setup and operation of the BEVFusion framework.




