# Task 1: Exporting a Model from PyTorch to ONNX

<details>
  <summary>1. Purpose</summary>
  Hướng dẫn cách xuất một mô hình PyTorch sang định dạng ONNX để sử dụng với ONNX Runtime.
</details>

<details>
  <summary>2. Action</summary>
  
  - Tải mô hình PyTorch có sẵn hoặc tự tạo một mô hình đơn giản.
  - Dùng `torch.onnx.export` để chuyển đổi mô hình sang định dạng ONNX.
  - Lưu mô hình ONNX.

  **Hình minh họa:**  
  ![Mô hình PyTorch](Export%20a%20PyTorch%20model%20to%20ONNX/pic.jpg)
</details>

<details>
  <summary>3. Result</summary>
  - Mô hình PyTorch đã được chuyển đổi thành `model.onnx`.
  - Kiểm tra lại mô hình bằng cách sử dụng `onnx.checker`.
</details>

<details>
  <summary>4. Upcoming</summary>
  - Kiểm tra mô hình ONNX với ONNX Runtime.
  - So sánh tốc độ suy luận giữa PyTorch và ONNX Runtime.
</details>

---

# Task 2: Running ONNX Model with ONNX Runtime

<details>
  <summary>1. Purpose</summary>
  Hướng dẫn cách chạy mô hình ONNX bằng ONNX Runtime để tối ưu hóa hiệu suất suy luận.
</details>

<details>
  <summary>2. Action</summary>

  - Cài đặt `onnxruntime`.
  - Load mô hình ONNX và chạy suy luận.
  - So sánh kết quả với mô hình PyTorch.

  **Hình minh họa:**  
  ![Ảnh gốc]((optional)%20Exporting%20a%20Model%20from%20PyTorch%20to%20ONNX%20and%20Running%20it%20using%20ONNX%20Runtime/cat_resized.jpg)  
  ![Ảnh sau khi xử lý với ONNX Runtime]((optional)%20Exporting%20a%20Model%20from%20PyTorch%20to%20ONNX%20and%20Running%20it%20using%20ONNX%20Runtime/cat_superres_with_ort.jpg)

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
