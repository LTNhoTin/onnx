import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import torchvision.transforms as transforms

# Kết nối với Triton Server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Load ảnh và tiền xử lý: Chuyển sang YCbCr và lấy kênh Y (Grayscale)
img = Image.open("test.jpg").convert("YCbCr")
img_y, img_cb, img_cr = img.split()[0], img.split()[1], img.split()[2]  # Lấy 3 kênh

# Resize ảnh về 224x224 và chuyển đổi sang tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_y = transform(img_y).unsqueeze(0).numpy()  # Shape: [1, 1, 224, 224]

# Định nghĩa input & output cho Triton
inputs = httpclient.InferInput("input", img_y.shape, "FP32")
inputs.set_data_from_numpy(img_y, binary_data=True)

outputs = httpclient.InferRequestedOutput("output", binary_data=True)

# Gửi request inference
results = client.infer(model_name="super_resolution", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy("output")  # Shape: [1, 1, ?, ?]

# Hậu xử lý: Chuyển output thành ảnh Grayscale
img_out_y = Image.fromarray(np.uint8((inference_output[0] * 255.0).clip(0, 255)[0]), mode='L')

# Ghép lại thành ảnh RGB với các kênh Cb và Cr đã resize
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]
).convert("RGB")

# Lưu ảnh kết quả
final_img.save("output_super_resolution.jpg")
print("Inference hoàn tất! Ảnh đã được tên output_super_resolution.jpg")
