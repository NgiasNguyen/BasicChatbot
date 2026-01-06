# Chatbot Hỏi-Đáp - Hệ thống Hỏi-Đáp Tiếng Việt

Dự án chatbot hỏi-đáp tiếng Việt với **2 phương pháp tìm kiếm**: TF-IDF và Semantic Search, cho phép so sánh hiệu quả giữa các phương pháp khác nhau.

## 🎯 Tính năng

### Chatbot TF-IDF (Nhanh)
- ✅ Đọc dữ liệu từ file CSV (định dạng: "question","answer")
- ✅ Tiền xử lý tiếng Việt: lowercase, bỏ dấu, chuẩn hóa khoảng trắng
- ✅ Vector hóa câu hỏi bằng TF-IDF
- ✅ Tìm câu trả lời phù hợp nhất sử dụng Cosine Similarity
- ⚡ **Ưu điểm**: Nhanh, hiệu quả, không cần GPU

### Chatbot Semantic Search (Chính xác)
- ✅ Sử dụng mô hình đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2`
- ✅ Hiểu ngữ nghĩa câu hỏi, không chỉ dựa trên từ khóa
- ✅ Trả lời dựa trên độ tin cậy (Confidence-based):
  - Score ≥ 0.75: Trả lời trực tiếp + % độ tin cậy
  - 0.45 < Score < 0.75: Hỏi lại + hiển thị câu trả lời
  - Score ≤ 0.45: Xin lỗi và yêu cầu diễn đạt lại
- 🧠 **Ưu điểm**: Chính xác hơn, hiểu ngữ cảnh

### Giao diện Streamlit
- ✅ Lựa chọn giữa 2 loại chatbot
- ✅ Giao diện chat đơn giản, trực quan
- ✅ Lịch sử chat tự động reset khi đổi loại chatbot

## 📦 Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python3.11 -m venv venv
source venv/bin/activate  # Trên macOS/Linux
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: 
- Python 3.11 được khuyến nghị (để hỗ trợ `torch` và `sentence-transformers`)
- Lần đầu chạy Semantic Search sẽ tải mô hình (~420MB), mất 2-5 phút

## 📁 Cấu trúc dự án

```
ChatbotQA/
├── chatbot.py              # Class Chatbot TF-IDF (bao gồm hàm tiền xử lý tiếng Việt)
├── chatbot_pro.py          # Class ChatbotPro Semantic Search
├── app.py                  # Giao diện Streamlit với lựa chọn 2 loại chatbot
├── data_converted.csv      # Dữ liệu câu hỏi-đáp (199 cặp)
├── requirements.txt        # Dependencies
├── README.md              # Tài liệu này
└── venv/                  # Môi trường ảo (không commit)
```

## 🚀 Cách sử dụng

### Chạy giao diện Streamlit (Khuyến nghị)

```bash
source venv/bin/activate
streamlit run app.py
```

Sau đó mở trình duyệt tại `http://localhost:8501`

**Cách sử dụng:**
1. Ở sidebar bên trái, chọn loại chatbot:
   - **TF-IDF (Nhanh)**: Dựa trên từ khóa, phù hợp FAQ đơn giản
   - **Semantic Search (Chính xác)**: Hiểu ngữ nghĩa, chính xác hơn
2. Nhập câu hỏi vào ô chat ở cuối trang
3. Xem kết quả và so sánh 2 phương pháp

### Chạy từ command line

**TF-IDF:**
```bash
python3 chatbot.py
```

**Semantic Search:**
```bash
python3 chatbot_pro.py
```

## 🔧 Cách hoạt động

### TF-IDF (chatbot.py)

1. **Load dữ liệu**: Đọc file CSV chứa các cặp câu hỏi-đáp
2. **Tiền xử lý**: 
   - Chuyển thành chữ thường
   - Bỏ dấu tiếng Việt
   - Chuẩn hóa khoảng trắng
3. **Vector hóa**: Sử dụng TF-IDF để chuyển đổi câu hỏi thành vector số
4. **Tìm kiếm**: 
   - Vector hóa câu hỏi người dùng bằng TF-IDF
   - Tính Cosine Similarity với tất cả câu hỏi trong database
   - Trả về câu trả lời có similarity cao nhất

### Semantic Search (chatbot_pro.py)

1. **Load mô hình**: Tải mô hình `paraphrase-multilingual-MiniLM-L12-v2`
2. **Load dữ liệu**: Đọc file CSV
3. **Encode**: Chuyển đổi tất cả câu hỏi mẫu thành embeddings (một lần khi khởi động)
4. **Tìm kiếm**:
   - Encode câu hỏi người dùng thành embedding
   - Tính Cosine Similarity với embeddings của câu hỏi mẫu
   - Trả lời dựa trên độ tin cậy (confidence-based)

## ⚙️ Tham số cấu hình

### TF-IDF (chatbot.py)

- `similarity_threshold`: Ngưỡng similarity tối thiểu (mặc định: 0.1)
- `ngram_range`: Phạm vi n-gram cho TF-IDF (mặc định: (1, 2))
- `max_features`: Số lượng features tối đa (mặc định: 5000)

### Semantic Search (chatbot_pro.py)

- `model_name`: Tên mô hình sentence-transformers (mặc định: 'paraphrase-multilingual-MiniLM-L12-v2')
- Ngưỡng độ tin cậy:
  - **Cao** (≥ 0.75): Trả lời trực tiếp
  - **Trung bình** (0.45 - 0.75): Hỏi lại + trả lời
  - **Thấp** (≤ 0.45): Xin lỗi

## 💡 Ví dụ sử dụng

### TF-IDF

```python
from chatbot import Chatbot

chatbot = Chatbot(csv_file='data_converted.csv', similarity_threshold=0.1)
chatbot.load_data()
chatbot.train()

answer = chatbot.answer("AI là gì?")
print(answer)
```

### Semantic Search

```python
from chatbot_pro import ChatbotPro

chatbot_pro = ChatbotPro(csv_file='data_converted.csv')
chatbot_pro.initialize()

answer, score, matched = chatbot_pro.answer("AI là gì?")
print(answer)
```

## 📋 Yêu cầu hệ thống

- **Python**: 3.11+ (khuyến nghị cho torch và sentence-transformers)
- **Dependencies**:
  - scikit-learn >= 1.0.0
  - numpy >= 1.21.0, < 2.0 (tương thích với torch)
  - streamlit >= 1.28.0
  - sentence-transformers >= 5.0.0
  - torch >= 2.0.0
  - pandas >= 2.0.0

## 📊 So sánh 2 phương pháp

| Tiêu chí | TF-IDF | Semantic Search |
|----------|--------|-----------------|
| **Tốc độ** | ⚡ Rất nhanh | 🐢 Chậm hơn (cần encode) |
| **Độ chính xác** | 📊 Trung bình | 🎯 Cao hơn |
| **Hiểu ngữ nghĩa** | ❌ Không | ✅ Có |
| **Yêu cầu GPU** | ❌ Không | ⚠️ Tùy chọn (CPU cũng được) |
| **Kích thước mô hình** | - | ~420MB |
| **Phù hợp** | FAQ đơn giản | Câu hỏi phức tạp, đa dạng |

## 🎓 Tác giả

Dự án được xây dựng cho mục đích học tập và nghiên cứu.

## 📝 Ghi chú

- Lần đầu chạy Semantic Search sẽ tải mô hình từ Hugging Face (cần internet)
- Mô hình được cache tự động, các lần sau sẽ nhanh hơn
- Dữ liệu mẫu: 199 cặp câu hỏi-đáp về AI và Chatbot
