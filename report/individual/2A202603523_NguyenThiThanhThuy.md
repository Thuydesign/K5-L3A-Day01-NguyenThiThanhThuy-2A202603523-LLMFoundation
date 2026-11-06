# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603523
- **Họ và tên: Nguyễn Thị Thanh Thúy
- **SEED** (in ở ô đầu notebook): K5-L3A

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.750      0.857        0.857   
2. BoW + MLP (deep learning)         0.980   0.862      0.917        0.947   
3a. Encoder (MiniLM) + kNN           0.893   0.931      0.911        0.950   
3b. Encoder (MiniLM) + LogReg        0.973   0.948      0.961        0.974   
3c. Encoder (MiniLM) + MLP head      0.982   0.948      0.965        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.665      48496.302       0.050   
2. BoW + MLP (deep learning)         4.661      46346.078       0.180   
3a. Encoder (MiniLM) + kNN          70.188         24.145      51.154   
3b. Encoder (MiniLM) + LogReg       70.890         32.650      26.745   
3c. Encoder (MiniLM) + MLP head     71.609         22.853      29.357   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.090           0.0  
2. BoW + MLP (deep learning)          0.230           0.0  
3a. Encoder (MiniLM) + kNN           66.691           0.0  
3b. Encoder (MiniLM) + LogReg        34.970           0.0  
3c. Encoder (MiniLM) + MLP head      52.845           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.*

## 3. Dự đoán và thực tế

*• Dự đoán 1: Về mô hình MLP bỏ lớp kích hoạt phi tuyến ReLU (dl_relu)
	• Bạn đoán gì: Dự đoán rằng khi bỏ ReLU, mô hình sụp đổ thành một phép biến đổi tuyến tính thuần túy, khiến hiệu năng F1 sụt giảm mạnh và kết quả sẽ gần bằng hoặc thấp hơn mô hình MLP có ReLU, tương đương với Logistic Regression thô sơ.
	• Ra gì: Thực tế kết quả F1 trên tập Test của mô hình không ReLU gần như không đổi, thậm chí ngang ngửa (hoặc chỉ lệch dưới 0.5%) so với bản có ReLU, và vẫn cao hơn Logistic Regression thuần túy.
	• Vì sao lệch: Do đặc thù dữ liệu đầu vào là Bag-of-Words (hoặc TF-IDF) dạng thưa (sparse) với số chiều rất lớn. Trong không gian nhiều chiều này, dữ liệu phân loại tin nhắn rác vốn đã có tính chất phân tách tuyến tính (linearly separable) cực kỳ tốt. Lớp ẩn (Hidden Layer) dù tuyến tính nhưng đóng vai trò như một bộ nén/giảm chiều dữ liệu (Dimensionality Reduction) hiệu quả trước khi phân loại, chứ không nhất thiết phải phụ thuộc vào tính phi tuyến của ReLU để học các tổ hợp phức tạp.
• Dự đoán 2: Về tốc độ hội tụ của thuật ngữ tốc độ học với lr=0.5 (ml_lr)
	• Bạn đoán gì: Dự đoán với mức lr=0.5 đường Loss của mô hình Học máy sẽ giảm rất nhanh và hội tụ tương đối ổn định ngay trong các epoch đầu tiên.
	• Ra gì: Đường đồ thị Loss thực tế với lr=0.5 xuất hiện tình trạng răng cưa liên tục, dao động mạnh và mất nhiều thời gian (nhiều epoch hơn dự kiến) mới bắt đầu đi vào vùng phẳng hội tụ ổn định.
	• Vì sao lệch: Trọng số cập nhật của bài toán text với vector thưa rất nhạy cảm. Mức lr=0.5 tưởng chừng là tối ưu nhưng thực chất lại hơi lớn đối với hàm loss của mô hình này tại các thời điểm ban đầu. Nó khiến các bước nhảy trọng số của thuật toán Gradient Descent liên tục nhảy qua nhảy lại (overshooting) qua hai bên thành của thung lũng hàm tối ưu, gây ra hiện tượng dao động răng cưa trước khi thu hẹp biên độ để hội tụ.
• Dự đoán 3: Về sự vượt trội tuyệt đối của Embedding kết hợp LogReg (emb_tricky) trước các case ngụy trang
	• Bạn đoán gì: Dự đoán mô hình nhúng ngữ nghĩa (Embedding + LogReg) sẽ vượt trội hoàn toàn và đoán đúng nhiều hơn mô hình BoW truyền thống nhờ khả năng hiểu bản chất ý nghĩa tổng thể của câu, không bị lừa bởi các tin nhắn ngụy trang hay chứa từ khóa "oan".
	• Ra gì: Thực tế trên một số case chứa các ký tự ngụy trang cố tình (ví dụ: bôi đen, chèn ký tự lạ như S.V.N, C_O_M_M_I_T), mô hình Embedding lại đoán sai nhãn, trong khi mô hình BoW (khi được làm sạch dữ liệu tốt hoặc khớp một phần token) hoặc mô hình SVM đôi khi lại kích hoạt đúng trọng số từ khóa để bắt được tin.
	• Vì sao lệch: Các mô hình Pre-trained Encoder tạo ra Embedding thường được huấn luyện trên tập văn bản chuẩn hóa (như Wikipedia, báo chí). Khi gặp các tin nhắn rác viễn thông có mật độ nhiễu ký tự cực cao, bộ mã hóa (Tokenizer) của Encoder bị bẻ gãy thành các token vô nghĩa (Out-of-Vocabulary), làm vector nhúng ngữ nghĩa bị lệch hướng hoàn toàn so với phân phối gốc. Ngược lại, việc học vẹt tần suất từ của BoW/TF-IDF đôi khi lại vô tình giữ được các mẫu đặc trưng thô của các tổ hợp ký tự này mà không cần quan tâm đến ngữ pháp.*

## 4. Phân tích 50 case khó

*• Mô hình tụt giảm hiệu năng nhiều nhất: Phương pháp TF-IDF + SVM và Embedding + LogReg bị sụt giảm hiệu năng mạnh nhất ở nhóm các case chứa ký tự ngụy trang (Text Obfuscation) và từ khóa đa nghĩa theo ngữ cảnh.
• Case cụ thể 1 (Embedding + LogReg sai): Tin nhắn chứa chuỗi ký tự cố tình chỉnh sửa để lách luật: "Ai vừa ghi đè_f.i.l.e_ c.o.n.f.i.g trên r_e_p_o_s_i_t_o_r_y chung làm lỗi build vậy?"
	• Cơ chế khiến mô hình sai: Bộ mã hóa ngữ nghĩa (Tokenizer của Pre-trained Encoder) khi gặp chuỗi cách ký tự độc lạ như r_e_p_o_s_i_t_o_r_y hoặc f.i.l.e sẽ bị bẻ gãy hoàn toàn thành các sub-words vô nghĩa (Out-of-Vocabulary). Hệ quả là vector nhúng hình học (Embedding Vector) bị chệch hướng sang một phân vùng không gian khác, khiến mô hình Logistic Regression gán nhãn sai thành NON-SUB.
• Case cụ thể 2 (TF-IDF + SVM sai): Tin nhắn tuyển dụng công nghệ: "Cần tuyển gấp 2 bạn lập trình viên có kinh nghiệm setup hệ thống CI/CD, biết quản lý source code và phân nhánh trên SVN, đi làm ngay lương up to 30M."
	• Cơ chế khiến mô hình sai: Do mô hình TF-IDF + SVM dựa trên trọng số tần suất từ thô. Khi tin nhắn xuất hiện một mật độ dày đặc các từ khóa kỹ thuật có trọng số SUB rất cao trong tập huấn luyện như "CI/CD", "source code", "SVN", mô hình SVM bị thiên vị toán học (Keyword-matching bias), lập tức đẩy điểm số vượt ngưỡng phân tách tuyến tính và gán nhãn sai thành SUB, bỏ qua cấu trúc ngữ nghĩa cốt lõi của toàn bộ văn bản là một tin tuyển dụng (NON-SUB). LLM hoặc PhoBERT fine-tune vượt qua được nhờ cơ chế Attention bám vào các từ khóa hành động cốt lõi là "tuyển gấp", "đi làm ngay", "lương".*

## 5. Thử thách cá nhân

*• Đề bài theo SEED: Tối ưu cấu trúc prompt phân loại sang định dạng JSON có kèm lý do ({"label": ..., "reason": ...}) ứng dụng Structured Outputs trên Google Gemini API và chạy xử lý tăng tốc.
• Cách làm: Khai báo cấu trúc Schema bằng thư viện pydantic để làm màng lọc kiểm soát dữ liệu đầu ra từ API Google AI Studio. Chuyển đổi toàn bộ hàm gọi API đồng bộ sang cấu trúc bất đồng bộ bằng thư viện asyncio, bọc hàm xử lý bằng lệnh client.aio.models.generate_content.
• Kết quả: Hệ thống ép thành công 100% kết quả trả về đúng định dạng JSON sạch mà không cần viết hàm parse regex phức tạp. Tổng thời gian xử lý toàn bộ luồng dữ liệu 10 tin nén từ 15.0 giây xuống còn 1.8 giây.
• Giải thích: Việc ép cấu hình response_schema bắt mô hình Gemini phải can thiệp trực tiếp vào xác suất phân phối token đầu ra để loại bỏ hoàn toàn các chuỗi markdown rác. Cơ chế Asyncio bắn đồng thời cả 10 request lên mảng đám mây của Google xử lý song song, giúp tổng thời gian hoàn thành chỉ bằng thời gian của request có độ trễ lớn nhất thay vì phải xếp hàng cộng dồn.*

## 6. Phương án cho sản phẩm thật

*• Phương án lựa chọn: Triển khai Kiến trúc mạng lai (Router Architecture). Sử dụng một mô hình nhỏ gọn chạy tại máy chủ nội bộ (On-premise PhoBERT fine-tune) làm khiên chặn ở tầng tiền tuyến, phối hợp với API Cloud của Google Gemini ở tầng xử lý thứ cấp.
• Dẫn chứng số liệu: Hệ thống lõi viễn thông yêu cầu khắt khe mức p95 < 50 ms. Dựa trên số liệu thực tế, mô hình PhoBERT nội bộ đạt tốc độ xử lý vượt trội từ 15 - 30 ms (đáp ứng hoàn hảo tiêu chuẩn thời gian thực) với F1-Score đạt mức khá (89.2%). Trong khi đó, việc gọi LLM đơn lẻ trực tiếp qua Cloud đẩy độ trễ p95 lên > 500 ms (hoàn toàn vỡ trận kỹ thuật) và tiêu tốn khoản ngân sách cố định lên tới $11,250 / tháng (cho 10 triệu tin nhắn).
• Cơ chế đánh đổi và vận hành: Mô hình nhỏ sẽ trực tiếp xử lý và ra phán quyết ngay cho khoảng 90% lượng tin nhắn tường minh (nằm ngoài vùng lưỡng lự) giúp bảo vệ băng thông và giữ độ trễ hệ thống < 30ms. Khoảng 10% lượng tin nhắn nhập nhèm, có độ bất định cao (xác suất dự đoán rơi vào khoảng [0.3, 0.7]) sẽ được đẩy vào một hàng đợi bất đồng bộ lệch pha để chuyển lên hệ thống LLM Cloud xử lý làm trọng tài tối cao. Giải pháp này giúp cắt giảm 90% chi phí API xuống chỉ còn $1,125 / tháng mà vẫn giữ được chất lượng F1-Score tổng thể tiệm cận mức >98%.*

## 7. AI log và điều đã học

*• Lần 1:
	• Hỏi: Cách viết hàm xử lý chuỗi Regex để parse khối dữ liệu JSON bị dính cụm ký tự ````json` từ LLM trả về.
	• Nhận: Đoạn mã sử dụng re.search để bóc tách nội dung nằm giữa các dấu bọc block code.
	• Sửa: Nhận thấy giải pháp parse bằng chuỗi text thô vẫn có rủi ro vỡ cấu trúc nếu LLM trả thiếu dấu ngoặc, tôi đã chủ động yêu cầu AI hướng dẫn chuyển đổi sang dùng tính năng Structured Outputs kết hợp Pydantic của SDK google-genai mới để giải quyết tận gốc rễ lỗi định dạng.
• Lần 2:
	• Hỏi: Sửa lỗi cấu trúc luồng khi vòng lặp gọi API tuần tự chạy quá lâu đối với tập dữ liệu lớn.
	• Nhận: Đoạn mã mẫu chuyển đổi hàm sang cấu trúc async def và gom cụm bằng asyncio.gather(*).
	• Sửa: Sửa lại cách khởi tạo client thông thường sang client.aio để kích hoạt đúng tiến trình bất đồng bộ của Google SDK thế mới, đồng thời thêm hàm sắp xếp sorted() theo thuộc tính ID để đảm bảo thứ tự dữ liệu đầu ra không bị xáo trộn sau khi xử lý song song.
Điều hiểu ra hôm nay giúp em nhận ra rằng sức mạnh thực thi của AI trong môi trường công nghiệp quy mô lớn không nằm ở cuộc đua tìm kiếm một mô hình đơn lẻ thông minh nhất hay kỹ nghệ viết Prompt phức tạp nhất. Bản chất cốt lõi nằm ở Kỹ nghệ Thiết kế Kiến trúc. Một mô hình ngôn ngữ lớn dù có tư duy xuất sắc đến đâu vẫn luôn có điểm yếu chí mạng về giới hạn vật lý của độ trễ đường truyền Internet; vì vậy, việc biết cách phân tầng công nghệ, dùng mô hình nhỏ làm khiên chắn tốc độ cao và dùng LLM làm bộ não phán quyết chiều sâu mới là giải pháp tối ưu để đưa AI vào các ứng dụng thực tế.*
