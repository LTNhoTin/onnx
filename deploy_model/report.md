# Báo cáo triển khai và đánh giá hiệu suất Model Super Resolution trên Triton

## **1️ Giới thiệu**
- **Mô hình**: Super Resolution (PyTorch → ONNX)
- **Triển khai**: Triton Inference Server
- **Inference trên**: CPU (MacBook Pro M1, Docker)
- **Mục tiêu**: Đánh giá hiệu suất của mô hình qua các tham số:
  - Throughput (ảnh xử lý mỗi giây)
  - Latency (thời gian xử lý mỗi request)
  - Ảnh hưởng của batch size & concurrency đến hiệu suất

---

## **2️ Cấu hình môi trường**
- **Môi trường cài đặt**:  
  - Python 3.10, PyTorch, ONNX, Triton Client, Docker
- **Mô hình sử dụng**:
  - **Mô hình Super Resolution đã được huấn luyện trong Training 1**.
  - **Định dạng xuất:** PyTorch → ONNX.
  - **Input:** `1 × 224 × 224` (Grayscale, YCbCr-Y Channel).
  - **Output:** `1 × 672 × 672` (Upscaled).
  - **Batch size hỗ trợ:** Lên đến 16.
---

## **3️ Kết quả đo hiệu suất**
### ** 3.1. Hiệu suất với Batch Size = 1 và 8**
| Chỉ số          | Batch Size = 1 | Batch Size = 8 |
|-----------------|----------------|----------------|
| **Throughput**  | 5.76 infer/sec | 5.32 infer/sec |
| **Avg Latency** | 171.9 ms       | 1420.4 ms      |
| **p50 Latency** | 170.3 ms       | 1417.9 ms      |
| **p95 Latency** | 172.9 ms       | 1444.9 ms      |
| **p99 Latency** | 193.7 ms       | 1468.2 ms      |

 **Nhận xét**:
- **Batch size lớn giúp giảm độ trễ trung bình trên mỗi ảnh** nhưng throughput không cải thiện nhiều.
- **Trên CPU, batch size lớn không giúp tăng tốc** → Trên GPU có thể sẽ tối ưu hơn.

---

### ** 3.2. Hiệu suất với nhiều mức Concurrency**
| **Concurrency** | **Throughput (infer/sec)** | **Avg Latency (ms)** |
|---------------|------------------|------------------|
| **1**         | 5.65 infer/sec   | 175 ms           |
| **3**         | 5.88 infer/sec   | 502 ms           |
| **5**         | 5.82 infer/sec   | 838 ms           |
| **7**         | 5.99 infer/sec   | 1.16 s           |
| **9**         | 5.87 infer/sec   | 1.5 s            |
| **11**        | ❌ Không ổn định  | ❌ Lỗi đo lường   |

 **Nhận xét**:
- **Concurrency = 3-5 có hiệu suất tốt nhất trên CPU**.
- **Latency tăng rất mạnh khi concurrency > 7**.
- **Concurrency > 10 không ổn định do CPU không đủ tài nguyên**.

---

## **4️ Tổng kết**
### ** Kết quả chính**
**Mô hình Super Resolution đã deploy thành công trên Triton.**  
**Inference trên CPU đạt tốc độ ~5.76 ảnh/giây với batch size = 1.**  
**Concurrency tối ưu nhất trên CPU: 3-5 request song song.**  
**Batch size lớn giúp giảm latency nhưng không cải thiện throughput trên CPU.**  

---
