# Triển khai Model Super Resolution trên Triton

## 1 Chuẩn bị Mô hình Super Resolution
### Mô hình
- Model Super Resolution đã được huấn luyện trong **Training 1**.
- Chuyển đổi từ **PyTorch → ONNX** để chạy trên Triton.

### Xuất Model từ PyTorch sang ONNX
1. Tạo model Super Resolution và tải trọng số đã huấn luyện trước.
2. Xuất model sang ONNX với dynamic batch:
   ```python
   torch.onnx.export(model, x, "super_resolution.onnx", export_params=True, opset_version=10,
                     do_constant_folding=True, input_names=['input'], output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
   ```
3. Kiểm tra lại model ONNX bằng ONNX Runtime:
   ```python
   import onnx
   onnx_model = onnx.load("super_resolution.onnx")
   onnx.checker.check_model(onnx_model)
   ```

---

## 2 Triển khai Model trên Triton
### Cài đặt Triton Server
1. Tải Triton Server từ NVIDIA Container Registry:
   ```sh
   docker pull --platform linux/amd64 nvcr.io/nvidia/tritonserver:25.01-py3
   ```

2. Tạo **model repository** và đặt model vào đúng thư mục:
   ```sh
   mkdir -p model_repository/super_resolution/1
   mv super_resolution.onnx model_repository/super_resolution/1/
   ```

3. **Tạo file `config.pbtxt`** để khai báo model trong Triton.

---

### Chạy Triton Server
```sh
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002     -v $(pwd)/model_repository:/models     --platform linux/amd64 nvcr.io/nvidia/tritonserver:25.01-py3     tritonserver --model-repository=/models
```
**Nếu server chạy thành công**, bạn sẽ thấy log:
```
+------------------+------+
| Model            | Version | Status |
+------------------+------+
| super_resolution | 1       | READY  |
```
**Model đã sẵn sàng để chạy inference!**

---

## 3 Kiểm tra Inference với Triton
### Cài đặt Triton Client
```sh
pip install tritonclient[http]
```

### Gửi request inference đến Triton
Chạy script `inference.py` để test inference:
```sh
python3 inference.py
```

### Kiểm tra kết quả
- Ảnh đầu vào: **`test.jpg`**
- Ảnh đầu ra được lưu: **`output_super_resolution.jpg`**
- Nếu inference thành công, ảnh được phóng to sẽ hiển thị.

---

## 4 Đo Hiệu Suất bằng Perf Analyzer
### Tải & Chạy Triton SDK
1. Tải Triton SDK:
   ```sh
   docker pull --platform linux/amd64 nvcr.io/nvidia/tritonserver:25.01-py3-sdk
   ```
2. Chạy SDK để đo hiệu suất:
   ```sh
   docker run --rm -it --net host --platform linux/amd64 nvcr.io/nvidia/tritonserver:25.01-py3-sdk
   ```

---

### Đo hiệu suất với Perf Analyzer
1. **Batch size = 1**:
   ```sh
   perf_analyzer -m super_resolution
   ```
2. **Batch size = 8**:
   ```sh
   perf_analyzer -m super_resolution -b 8
   ```
3. **Thử nghiệm với nhiều concurrency (1-11)**:
   ```sh
   perf_analyzer -m super_resolution --concurrency-range 1:11:2
   ```
🔹 **Thông số quan trọng được đo**:
- **Throughput** (ảnh/giây)
- **Latency** (thời gian xử lý mỗi request)
- **Ảnh hưởng của batch size & concurrency đến hiệu suất**

**Sau khi đo xong, tổng hợp kết quả vào báo cáo `report.md`.**