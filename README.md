# Lab 02 — Building and Training Neural Networks with Scikit-Learn & PyTorch

**Môn học:** Machine Learning in Data Science
**Sinh viên:** Phạm Tiến Triển
**MSSV:** SE182026

Toàn bộ code, kết quả và phần nhận xét nằm trong notebook **`Lab2.ipynb`**. 

---

## 1. Môi trường thực hiện

| Thành phần | Phiên bản / cấu hình |
|---|---|
| Python | 3.13.9 (Anaconda) |
| PyTorch | 2.14.1 + CUDA 13.0 |
| torchvision | 0.29.1 |
| scikit-learn, NumPy, pandas, Matplotlib, Seaborn | bản đi kèm Anaconda |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| Môi trường chạy | VS Code + Jupyter, Windows |

Phần 1 và 2 chạy trên CPU; Phần 3 chạy trên GPU (code tự chuyển sang CPU nếu không có GPU, nhưng thời gian huấn luyện sẽ lâu hơn nhiều).

---

## 2. Cách chạy lại notebook

1. Cài thư viện (nếu chưa có PyTorch bản GPU):
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
2. Mở `Lab2.ipynb` và chạy lần lượt các cell **từ trên xuống dưới** (Run All). Các phần sau dùng lại biến của các phần trước.
3. Bộ dữ liệu CIFAR-10 (~170 MB) được tự động tải về thư mục `./data/` ở mục 3.2.1 trong lần chạy đầu tiên.
4. Mọi thí nghiệm đều cố định seed (`random_state=42` cho scikit-learn, `torch.manual_seed(42)` cho PyTorch) để kết quả có thể lặp lại. Kết quả trên GPU có thể chênh lệch rất nhỏ giữa các lần chạy do tính toán song song.

**Thời gian chạy ước tính** (với cấu hình ở mục 1, dữ liệu đã có sẵn):

| Phần | Thời gian |
|---|---|
| 1.3 — GridSearchCV (24 tổ hợp × 5-fold = 120 lần huấn luyện) | ~16 phút |
| 3.2.3 — Huấn luyện CustomCNN (30 epoch) | ~6 phút |
| 3.3.2 — Huấn luyện ResNet-20 (30 epoch) | ~9 phút |
| Các phần còn lại | vài giây mỗi cell |
| **Tổng** | **~35 phút** |

---

## 3. Các file trong thư mục

| File | Nội dung |
|---|---|
| `Lab2.ipynb` | Bài làm: code, output và nhận xét cho tất cả các phần |
| `README.md` | File này |
| `data/` | Dữ liệu CIFAR-10, được tạo tự động khi chạy mục 3.2.1 |
| `custom_cnn.pt`, `resnet20.pt` | Trọng số của CustomCNN và ResNet-20 sau khi huấn luyện, được tạo tự động ở mục 3.2.3 và 3.3.2 |
| `custom_cnn_history.json`, `resnet20_history.json` | Loss và accuracy theo từng epoch của hai mô hình trên |

Các file `.pt` cho phép nạp lại mô hình đã huấn luyện mà không cần huấn luyện lại:
```python
resnet = ResNet20().to(device)
resnet.load_state_dict(torch.load("resnet20.pt", map_location=device))
```

---

## 4. Tóm tắt kết quả

### Part 1 — Regression MLP (California Housing)

| Mô hình | Test RMSE | Test R² |
|---|---|---|
| Baseline: `(64,)`, ReLU, Adam | 0.5428 | 0.7752 |
| Best GridSearch: `(128, 64)`, ReLU, lr = 0.001, alpha = 0.001 | **0.5137** | **0.7986** |

- Baseline hội tụ ở iteration 220/500; RMSE trên train (~0.53) và test (0.54) gần bằng nhau → không overfitting.
- GridSearch giảm RMSE 5,4%. CV RMSE trung bình theo độ sâu: 1 lớp 0.5500, 2 lớp 0.5252, 3 lớp 0.5495 → mạng 2 lớp tốt nhất; alpha ảnh hưởng nhỏ (0.5430 so với 0.5401).

### Part 2 — Classification MLP (Breast Cancer Wisconsin)

| Thí nghiệm | Kết quả trên tập test (114 mẫu) |
|---|---|
| 2.2 — MLP `(32, 16)`, ReLU, Adam | Accuracy 95,61%; lớp malignant: precision 0.911, recall 0.976, F1 0.943 |
| 2.3 — SGD / Adam / L-BFGS | Accuracy 98,25% / 95,61% / 94,74%; số iteration 331 / 295 / 16 |
| 2.4 — Không early stopping / Có early stopping | Accuracy 95,61% / 92,11%; số iteration 295 / 25 |

- 2.2: chỉ bỏ sót 1/42 ca ác tính, báo nhầm 4/72 ca lành tính.
- 2.3: L-BFGS hội tụ nhanh nhất và đạt loss thấp nhất, nhưng accuracy trên test thấp nhất → loss trên train thấp không đồng nghĩa với tổng quát hóa tốt.
- 2.4: early stopping tiết kiệm 92% số iteration nhưng giảm accuracy, do tập validation chỉ có 46 mẫu nên điểm validation bão hòa sớm và mô hình dừng trước khi học đủ.

### Part 3 — Deep Learning with PyTorch (CIFAR-10)

| Thí nghiệm | Kết quả |
|---|---|
| 3.1 — Nhân ma trận 2048 × 2048 | CPU 67,62 ms, GPU 2,27 ms → GPU nhanh hơn **29,7 lần** |
| 3.1 — Autograd, f(x) = (x² + 3x − 2)³ tại x = 2 | df/dx = **1344** (khớp với tính tay) |
| 3.3.1 — Gradient norm qua 21 lớp conv | Có skip: 1.17 – 2.13 ở mọi lớp; không skip: giảm còn 0.088, yếu hơn tới 23 lần |

| Mô hình | Số tham số | Thời gian huấn luyện | Train acc | Val acc | **Test acc** |
|---|---|---|---|---|---|
| CustomCNN (3.2) — Adam, StepLR | 175.658 | 6,2 phút | 81,63% | 81,66% | **81,97%** |
| ResNet-20 (3.3) — SGD, CosineAnnealingLR | 272.474 | 8,8 phút | 94,60% | 89,60% | **89,30%** |

- CustomCNN không overfitting (train và validation gần như trùng nhau); giới hạn nằm ở dung lượng mô hình.
- ResNet-20 cao hơn CustomCNN 7,3 điểm phần trăm (giảm ~40% số ảnh đoán sai), với số tham số nhiều hơn 55% và thời gian huấn luyện lâu hơn 42%; có overfitting nhẹ (chênh 5 điểm giữa train và validation) nhưng validation accuracy vẫn tăng đến epoch cuối.

---

## 5. Những điểm làm thêm ngoài yêu cầu của đề

Trong lúc làm, có một số chỗ em làm thêm hoặc làm khác một chút so với đề. Lý do của từng chỗ như sau:

1. **Chia dữ liệu ung thư vú cho đều (Part 2).** Khi chia train/test, em dùng `stratify` để tỉ lệ khối u lành tính và ác tính ở hai tập giống nhau (62,6% và 63,2%). Nếu không, tập test có thể bị lệch, ví dụ quá ít ca ác tính, làm kết quả đánh giá không chính xác.

2. **Sửa lỗi hết bộ nhớ khi chạy GridSearch (1.3).** Lần chạy đầu bị lỗi `MemoryError` vì máy mở nhiều tiến trình chạy song song cùng lúc, và mỗi tiến trình phải nạp thêm nhiều thứ không cần thiết nên hết RAM. Em sửa bằng cách thay đổi cách ẩn cảnh báo, để các tiến trình không phải nạp những thứ đó nữa. Nhờ vậy vẫn chạy song song được (`n_jobs=-1`) như đề yêu cầu.

3. **Tự vẽ đường loss cho L-BFGS (2.3).** scikit-learn chỉ lưu đường loss cho SGD và Adam, không lưu cho L-BFGS. Để vẽ đủ cả 3 đường trên cùng một hình như đề yêu cầu, em cho L-BFGS học lần lượt 1 bước, 2 bước, 3 bước, … rồi ghi lại loss sau mỗi lần. Cách này cho ra đúng đường loss thật, vì L-BFGS mỗi lần chạy đều cho kết quả giống hệt nhau.

4. **Tách thêm tập validation cho CIFAR-10 (3.2).** Đề yêu cầu vẽ đường loss và accuracy của validation, nhưng CIFAR-10 chỉ có sẵn tập train và tập test. Em không muốn dùng tập test trong lúc huấn luyện, vì tập test chỉ nên dùng một lần ở cuối để chấm điểm. Vì vậy em tách 5.000 ảnh từ tập train ra làm validation: còn 45.000 ảnh train, 5.000 ảnh validation, 10.000 ảnh test. Cả CustomCNN và ResNet-20 đều dùng cùng cách chia này để so sánh công bằng.

5. **Đo tốc độ GPU cho chính xác (3.1).** Lần chạy đầu tiên trên GPU luôn chậm vì GPU cần khởi động, nên em chạy thử một lần trước rồi mới bấm giờ, và lấy trung bình 10 lần. Ngoài ra, GPU làm việc song song với chương trình, nên em thêm lệnh chờ GPU làm xong (`torch.cuda.synchronize()`) rồi mới dừng đồng hồ; nếu không, thời gian đo được sẽ ngắn hơn thực tế.

6. **So sánh có và không có skip connection (3.3.1).** Đề yêu cầu in độ lớn gradient ở từng lớp. Để thấy rõ tác dụng của skip connection, em thêm một tùy chọn `use_skip` để tắt nó đi, rồi so sánh hai mạng giống hệt nhau (cùng trọng số ban đầu), một mạng có và một mạng không có skip connection.

7. **Lưu ý khi so sánh CustomCNN và ResNet-20 (3.3.2).** Theo đề, hai mô hình học theo hai cách khác nhau: CustomCNN dùng Adam + StepLR, còn ResNet-20 dùng SGD + CosineAnnealingLR. Vì vậy ResNet-20 tốt hơn có thể một phần do cách học, chứ không hoàn toàn do kiến trúc có skip connection.

8. **Lưu mô hình sau khi huấn luyện (3.2.3 và 3.3.2).** Em lưu mô hình đã học ra file `.pt` để nếu phải khởi động lại notebook thì không cần huấn luyện lại từ đầu (mỗi lần mất 6–9 phút).
