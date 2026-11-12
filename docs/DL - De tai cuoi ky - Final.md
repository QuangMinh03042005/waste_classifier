**Đề tài cuối kỳ**

**Học sâu**

# **ĐỀ TÀI CUỐI KỲ MÔN HỌC SÂU**

## **1. MỤC TIÊU**

Đề tài cuối kỳ giúp sinh viên vận dụng kiến thức của học phần Học sâu để xây dựng, huấn luyện và đánh giá mô hình cho một bài toán thực tiễn. Trọng tâm là phương pháp nghiên cứu, thiết kế thực nghiệm có kiểm soát và khả năng giải thích kết quả, thay vì chỉ đạt điểm số cao.

- Xác định bài toán, mục tiêu và 1–3 câu hỏi nghiên cứu (Research Questions – RQ) cụ thể, có thể kiểm chứng bằng thực nghiệm.

- • Lựa chọn dữ liệu, xây dựng pipeline tiền xử lý, huấn luyện và đánh giá phù hợp với dữ liệu ảnh, chuỗi, văn bản hoặc dữ liệu có cấu trúc.

- Thiết kế và triển khai mô hình học sâu bằng Python và PyTorch; giải thích kiến trúc, hàm mất mát, tối ưu hóa và chiến lược huấn luyện.

- So sánh mô hình với baseline và thực hiện ablation studies để kiểm tra vai trò của ít nhất một thành phần hoặc lựa chọn thiết kế.

- Phân tích kết quả định lượng, định tính, lỗi dự đoán, khả năng tổng quát hóa và giới hạn nghiên cứu.

- Khuyến khích sử dụng kiến trúc tiên tiến, transfer learning và xây dựng demo ứng dụng thực tế bằng Streamlit, Gradio hoặc giải pháp tương đương.

## **2. HÌNH THỨC THỰC HIỆN**

- Nhóm tối đa 04 sinh viên. Khối lượng nghiên cứu, số thí nghiệm và chiều sâu phân tích cần tương xứng với số thành viên.

- Mỗi nhóm chọn một bài toán chính và xác lập RQ trước khi huấn luyện. Mỗi RQ phải gắn với giả thuyết, phép đối chứng, độ đo và kết quả minh chứng.

- Bắt buộc sử dụng PyTorch cho pipeline mô hình học sâu (định nghĩa/tải mô hình, training hoặc finetuning, validation, inference). Có thể dùng torchvision, torchaudio, torchtext, timm, Hugging Face Transformers hoặc PyTorch Lightning khi phù hợp.

- Tối thiểu 02 cấu hình/mô hình để so sánh, trong đó có ít nhất 01 baseline rõ ràng. Bắt buộc có ablation study với ít nhất 02 biến thể đối chứng ngoài mô hình đầy đủ; số lượng thí nghiệm phải hợp lý với tài nguyên.

- Không bắt buộc đề xuất thuật toán mới; ưu tiên cách đặt vấn đề, giả thuyết và kiểm chứng có giá trị.

**3. NỘI DUNG YÊU CẦU**

- **3.1. Xác định bài toán, câu hỏi nghiên cứu và dữ liệu**

- Trình bày bối cảnh, ý nghĩa, đối tượng sử dụng, input/output và phạm vi bài toán (phân lớp, hồi quy, nhận dạng, dự báo chuỗi, phân đoạn ảnh, phát hiện bất thường, xử lý ngôn ngữ...).

- Đề xuất 1–3 RQ đo lường được. Ví dụ: “Fine-tuning backbone pretrained có cải thiện Macro-F1 so với huấn luyện từ đầu trên cùng tập dữ liệu không?” hoặc “Attention có cải thiện kết quả so với mô hình bỏ attention không?”.

- Với mỗi RQ lập bảng: giả thuyết → baseline/biến thể đối chứng → dữ liệu/chia tập → metric → tiêu chí kết luận. Không đặt RQ chung chung như “mô hình nào tốt nhất?”.

- Nguồn dữ liệu phải hợp lệ, rõ giấy phép, kích thước, nhãn, phân bố lớp, chất lượng và giới hạn. Có thể dùng dữ liệu công khai hoặc tự thu thập có mô tả quy trình và quyền sử dụng.

### **3.2. Khám phá, tiền xử lý và chia dữ liệu**

- Thực hiện EDA và trực quan hóa đặc thù (ảnh mẫu, phân bố chiều dài văn bản/chuỗi, class imbalance, tỷ lệ dữ liệu thiếu, v.v.).


**Đề tài cuối kỳ**

**Học sâu**

- Mô tả resize/normalization, tokenization, sliding window, augmentation hoặc feature extraction; giải thích tác động dự kiến và áp dụng nhất quán.

- Phân chia train/validation/test phù hợp; ưu tiên chia theo người, thiết bị, bệnh nhân, tài liệu gốc hoặc theo thời gian khi cần; loại bỏ trùng lặp và kiểm soát data leakage.

- Mọi phép fit của preprocessing, lựa chọn mô hình và tối ưu siêu tham số chỉ sử dụng train/validation. Test set chỉ dùng đánh giá cuối cùng.

### **3.3. Xây dựng mô hình và huấn luyện bằng PyTorch**

- Ít nhất một mô hình học sâu phù hợp nội dung học phần: MLP, CNN, RNN/LSTM/GRU; có thể mở rộng ResNet, EfficientNet, ViT, Transformer, TCN, U-Net, mô hình pretrained phù hợp.

- Mô tả kiến trúc và sơ đồ pipeline, tensor input/output, layers chính, loss, optimizer, learning rate, batch size, epochs, scheduler, regularization và checkpoint.

- Ghi rõ huấn luyện từ đầu hay transfer learning/fine-tuning; sử dụng pretrained model phải chỉ ra phần được đóng băng/mở khóa và trọng số nguồn.

- Mã nguồn tổ chức thành phần dữ liệu, model, train, evaluate, inference; lưu seed, cấu hình và log huấn luyện. Khuyến khích mixed precision và early stopping khi phù hợp.

- **3.4. Thiết kế thực nghiệm nhằm trả lời RQ**

- Lập ma trận thí nghiệm trước khi chạy: tên thí nghiệm, thay đổi duy nhất, phần giữ cố định, metric, RQ liên quan.

- So sánh công bằng với cùng split, tiền xử lý cốt lõi, metric và ngân sách huấn luyện hợp lý; khi khác biệt tài nguyên phải nêu rõ.

- Thực hiện ablation studies bắt buộc: mô hình đầy đủ và ít nhất 02 biến thể lược bỏ/thay đổi một thành phần (ví dụ: bỏ augmentation; bỏ attention; thay pretrained bằng scratch; bỏ skip connection). Tránh gộp nhiều thay đổi khiến không thể kết luận.

- Khuyến khích chạy nhiều seed (ví dụ 3) cho thí nghiệm chính, báo cáo mean ± std; nếu hạn chế GPU, nêu rõ giới hạn và dùng đánh giá bổ sung phù hợp.

- Chọn metric theo bài toán: Macro-F1/Balanced Accuracy/AUROC cho phân lớp; MAE/RMSE cho hồi quy; mIoU/Dice cho segmentation; BLEU/ROUGE/BERTScore khi phù hợp với NLP. Báo cáo thời gian huấn luyện, latency, số tham số hoặc VRAM khi liên quan RQ.

### **3.5. Phân tích, diễn giải và thảo luận**

- Cung cấp bảng kết quả baseline, mô hình đề xuất và từng ablation; biểu đồ train/validation loss và metric theo epoch.

- Phân tích lỗi bằng confusion matrix, ví dụ sai/đúng, vùng ảnh ảnh hưởng (Grad-CAM), attention visualization hoặc phân tích theo nhóm dữ liệu khi phù hợp.

- Trả lời từng RQ bằng dữ liệu cụ thể: kết quả ủng hộ hay bác bỏ giả thuyết, mức chênh lệch, tính ổn định và yếu tố gây nhiễu. Không khẳng định quan hệ nhân quả nếu đối chứng chưa đủ.

- Thảo luận overfitting/underfitting, khả năng tổng quát hóa, độ phức tạp, thiên lệch dữ liệu và hạn chế của thí nghiệm.

### **3.6. Ứng dụng minh họa (khuyến khích)**

- Khuyến khích demo bằng Streamlit/Gradio/web app/API: nhận input thực tế, chạy inference bằng checkpoint PyTorch, trả prediction và mức tin cậy/hiển thị phù hợp.

- Nêu kịch bản sử dụng, hướng dẫn chạy và giới hạn ứng dụng. Demo chỉ mang tính minh họa, không thay thế thực nghiệm và kiểm chứng RQ.

- **3.7. Sản phẩm nộp**

- Báo cáo hoàn chỉnh theo cấu trúc Mục 4 (DOCX hay PDF); slide thuyết trình.


**Đề tài cuối kỳ**

**Học sâu**

- Mã nguồn .py/.ipynb; requirements.txt hoặc environment.yml; README với lệnh tái lập, seed, cách tải dữ liệu/trọng số và cấu trúc thư mục.

- Bảng kết quả thí nghiệm, logs, cấu hình, hình minh họa, checkpoint hoặc đường dẫn tải checkpoint hợp lệ; mô tả tài nguyên tính toán.

- Tài liệu nguồn dataset, bài báo/mô hình tham khảo và bảng phân công công việc, tỷ lệ đóng góp của từng thành viên.

- Nếu có demo: source code, hướng dẫn chạy, ảnh/video minh họa hoặc đường dẫn thử nghiệm.

## **4. CẤU TRÚC BÁO CÁO**

**Chương 1. Giới thiệu:** Bối cảnh, vấn đề, mục tiêu, RQ và giả thuyết; input/output; đóng góp và phạm vi.

**Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan:** Các kiến trúc DL liên quan, phương pháp đối chứng, nguồn mô hình pretrained/SOTA; lý do chọn giải pháp.

**Chương 3. Dữ liệu và phương pháp:** Dataset/EDA, preprocessing, split, sơ đồ kiến trúc, training config, baseline, metric và ma trận RQ–thí nghiệm; thiết kế ablation.

**Chương 4. Thực nghiệm, kết quả và thảo luận:** Môi trường thực nghiệm, bảng kết quả, biểu đồ, ablation, phân tích lỗi và trả lời từng RQ; minh họa ứng dụng nếu có.

**Chương 5. Kết luận và hướng phát triển:** Kết luận theo RQ, đóng góp, hạn chế và hướng mở rộng.

## **5. QUY ĐỊNH VỀ DỮ LIỆU VÀ TRUNG THỰC HỌC THUẬT**

- Phải trích dẫn dataset, bài báo, pretrained weights, mã nguồn, công cụ và nội dung được AI hỗ trợ. Không trình bày mô hình có sẵn như đóng góp do nhóm tự đề xuất.

- Không sao chép mã nguồn, báo cáo hoặc kết quả; không chỉnh sửa/chọn lọc kết quả để che giấu thí nghiệm bất lợi.

- Không dùng test set để chọn epoch, threshold, siêu tham số hoặc quyết định biến thể ablation. Không để ảnh/chuỗi/đối tượng liên quan xuất hiện đồng thời giữa các split gây leakage.

- Báo cáo trung thực giới hạn GPU, seed, số lần chạy, thất bại và khác biệt giữa reproduction và kết quả công bố.

## **6. RUBRIC ĐÁNH GIÁ (100 ĐIỂM)**

| **Tiêu chí**                           | **Mô tả**                                                                                 | **Điểm**<br>**tối đa** |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------- |
| 1. Bài toán, dữ liệu và<br>RQ          | Bài toán rõ ràng; RQ cụ thể, kiểm chứng được; dữ liệu hợp lệ, mô tả đầy<br>đủ.            | 15                     |
| 2. Dữ liệu và tiền xử lý               | EDAphù hợp;split đúng; preprocessingcó lýdo;kiểm soát leakage.                            | 10                     |
| 3. Thiết kế thực nghiệm                | Baseline,metric,ma trận RQ–thí nghiệm;đối chứngcôngbằng.                                  | 15                     |
| 4. Triển khai PyTorch<br>và huấn luyện | Kiến trúc phù hợp; cài đặt đúng; training/fine-tuning, config và source<br>có tổ chức.    | 20                     |
| 5. Ablation studies và<br>đánhgiá      | Ít nhất 02 biến thể ablation; kết quả định lượng rõ; phân tích độ ổn<br>định/tuninghợplý. | 20                     |
| 6. Phân tích và trả lời<br>RQ          | Diễn giải kết quả, failure cases, trả lời RQ có chứng cứ và chỉ ra hạn chế.               | 10                     |
| 7. Báo cáo và thuyết<br>trình          | Cấu trúc logic, hình/bảng rõ, trích dẫn, trình bày và phản biện.                          | 5                      |
| 8. Tái lậpvà trungthực                 | README,code chạylại,logs/cấu hình và nguồn hỗ trợminh bạch.                               | 5                      |
| TỔNG                                   |                                                                                           | 100                    |

Điểm khuyến khích (tối đa +10 điểm, điểm cuối cùng không vượt quá thang điểm quy định): ứng dụng Streamlit/Gradio có giá trị thực tế (+0–4); sử dụng hoặc tái lập SOTA có phân tích công bằng (+0–3); thực


**Đề tài cuối kỳ**

**Học sâu**

nghiệm bổ sung có giá trị (nhiều seed, robustness, efficiency, external test...) (+0–3). Không cộng điểm chỉ vì sử dụng mô hình lớn.

## **7. LƯU Ý VÀ VÍ DỤ THIẾT KẾ THỰC NGHIỆM**

- Ví dụ RQ1: “Transfer learning có cải thiện Macro-F1 so với huấn luyện CNN từ đầu không?” → Đối chứng CNN scratch với CNN pretrained, cùng split, metric và kế hoạch huấn luyện được mô tả.

- Ví dụ RQ2: “Augmentation và class-weighted loss góp phần như thế nào?” → Full (cả hai), bỏ augmentation, bỏ weighted loss; ghi rõ các yếu tố còn lại được giữ cố định.

- Ví dụ bảng kết quả: Experiment | Pretrained | Augmentation | Weighted loss | Macro-F1 | Latency | Seed(s).

- Một kết quả không cải thiện vẫn có giá trị nếu giả thuyết và phép kiểm chứng đúng. Mô hình SOTA là lựa chọn khuyến khích, không bắt buộc; mô hình quá lớn vượt năng lực GPU có thể dùng pretrained/finetuning có kiểm soát.

- Nội dung học phần chính là MLP, CNN và RNN/LSTM/GRU; kiến trúc Transformer/ViT hoặc mô hình mới là phần mở rộng tự chọn và phải giải thích được.

---o0o---

Hết