# ✅ Task 1: Exporting a Model from PyTorch to ONNX

<details>
  <summary>1. Purpose</summary>
  Hướng dẫn cách xuất một mô hình PyTorch sang định dạng ONNX để sử dụng với ONNX Runtime.
</details>

<details>
  <summary>2. Action</summary>
  
  - Tải mô hình PyTorch có sẵn hoặc tự tạo một mô hình đơn giản.
  - Dùng `torch.onnx.export` để chuyển đổi mô hình sang định dạng ONNX.
  - Lưu mô hình ONNX.

  **Mã nguồn:**
  ```python
  import torch
  import torch.onnx

  # Giả sử ta có một mô hình PyTorch
  model = MyPyTorchModel()
  model.eval()

  # Xuất mô hình sang ONNX
  dummy_input = torch.randn(1, 3, 224, 224)
  torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
  ```
</details>

<details>
  <summary>3. Result</summary>
  - Mô hình PyTorch được chuyển đổi thành `model.onnx`.
  - Kiểm tra lại mô hình bằng cách sử dụng `onnx.checker`.
</details>

<details>
  <summary>4. Upcoming</summary>
  - Kiểm tra mô hình ONNX với ONNX Runtime.
  - So sánh tốc độ suy luận giữa PyTorch và ONNX Runtime.
</details>

---

# ✅ Task 2: Running ONNX Model with ONNX Runtime

<details>
  <summary>1. Purpose</summary>
  Hướng dẫn cách chạy mô hình ONNX bằng ONNX Runtime để tối ưu hóa hiệu suất suy luận.
</details>

<details>
  <summary>2. Action</summary>

  - Cài đặt `onnxruntime`.
  - Load mô hình ONNX và chạy suy luận.
  - So sánh kết quả với mô hình PyTorch.

  **Mã nguồn:**
  ```python
  import onnxruntime as ort
  import numpy as np

  # Load mô hình ONNX
  session = ort.InferenceSession("model.onnx")

  # Chuẩn bị đầu vào
  input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

  # Thực hiện suy luận
  outputs = session.run(None, {"input": input_data})
  ```
</details>

<details>
  <summary>3. Result</summary>
  - Mô hình ONNX chạy thành công với ONNX Runtime.
  - Hiệu suất suy luận nhanh hơn so với PyTorch.
</details>

<details>
  <summary>4. Upcoming</summary>
  - Benchmark mô hình ONNX so với các phương pháp khác.
  - Triển khai trên các nền tảng nhúng hoặc đám mây.
</details>
