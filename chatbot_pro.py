#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Hỏi-Đáp sử dụng Semantic Search với sentence-transformers
Sử dụng mô hình đa ngôn ngữ để tìm kiếm câu trả lời dựa trên ngữ nghĩa
"""

import csv
from sentence_transformers import SentenceTransformer, util
import torch


class ChatbotPro:
    """
    Chatbot hỏi-đáp sử dụng Semantic Search với sentence-transformers
    """
    
    def __init__(self, csv_file='data_converted.csv', model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Khởi tạo ChatbotPro
        
        Args:
            csv_file: Đường dẫn đến file CSV chứa dữ liệu
            model_name: Tên mô hình sentence-transformers
        """
        self.csv_file = csv_file
        self.model_name = model_name
        self.model = None
        self.questions = []
        self.answers = []
        self.corpus_embeddings = None
        self.initialized = False
    
    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        questions = []
        answers = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    question = row['question'].strip()
                    answer = row['answer'].strip()
                    
                    if question and answer:
                        questions.append(question)
                        answers.append(answer)
            
            self.questions = questions
            self.answers = answers
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            return False
    
    def initialize(self):
        """Khởi tạo mô hình và encode dữ liệu"""
        if self.initialized:
            return True
        
        # Load mô hình
        self.model = SentenceTransformer(self.model_name)
        
        # Load dữ liệu
        if not self.load_data():
            return False
        
        if not self.questions:
            return False
        
        # Encode câu hỏi mẫu
        self.corpus_embeddings = self.model.encode(
            self.questions,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        self.initialized = True
        return True
    
    def answer(self, user_question):
        """
        Trả lời câu hỏi của người dùng
        
        Args:
            user_question: Câu hỏi của người dùng
            
        Returns:
            Tuple (answer_text, confidence_score, matched_question)
        """
        if not self.initialized:
            return "Chatbot chưa được khởi tạo", 0.0, ""
        
        # Encode câu hỏi người dùng
        query_embedding = self.model.encode(user_question, convert_to_tensor=True)
        
        # Tính cosine similarity
        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Tìm câu hỏi tốt nhất
        best_idx = cosine_scores.argmax().item()
        best_score = float(cosine_scores[best_idx])
        
        # Logic trả lời dựa trên độ tin cậy
        if best_score >= 0.75:
            # Độ tin cậy cao: Trả lời trực tiếp
            confidence_percent = best_score * 100
            answer_text = f"{self.answers[best_idx]}\n\n*(Độ tin cậy: {confidence_percent:.1f}%)*"
            return answer_text, best_score, self.questions[best_idx]
            
        elif best_score > 0.45:
            # Độ tin cậy trung bình: Hỏi lại kèm câu trả lời
            answer_text = f"Có phải ý bạn là: **\"{self.questions[best_idx]}\"**?\n\n**Trả lời:** {self.answers[best_idx]}\n\n*(Độ tương đồng: {best_score:.2f})*"
            return answer_text, best_score, self.questions[best_idx]
            
        else:
            # Độ tin cậy thấp: Xin lỗi
            answer_text = f"Xin lỗi, tôi chưa hiểu ý bạn, vui lòng diễn đạt lại.\n\n*(Độ tương đồng tốt nhất: {best_score:.2f})*"
            return answer_text, best_score, ""


def load_data(csv_file='data_converted.csv'):
    """
    Đọc dữ liệu câu hỏi-đáp từ file CSV
    
    Args:
        csv_file: Đường dẫn đến file CSV chứa dữ liệu
        
    Returns:
        questions: Danh sách câu hỏi mẫu
        answers: Danh sách câu trả lời tương ứng
    """
    questions = []
    answers = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                question = row['question'].strip()
                answer = row['answer'].strip()
                
                # Chỉ thêm các cặp câu hỏi-đáp hợp lệ
                if question and answer:
                    questions.append(question)
                    answers.append(answer)
        
        print(f"✓ Đã tải {len(questions)} cặp câu hỏi-đáp từ {csv_file}")
        return questions, answers
        
    except FileNotFoundError:
        print(f"✗ Không tìm thấy file: {csv_file}")
        return [], []
    except Exception as e:
        print(f"✗ Lỗi khi đọc file: {e}")
        return [], []


def main():
    """
    Hàm main: Khởi tạo chatbot và chạy vòng lặp tương tác
    """
    print("=" * 60)
    print("CHATBOT HỎI-ĐÁP - Semantic Search")
    print("=" * 60)
    print("\nĐang khởi tạo mô hình...")
    
    # Bước 1: Load mô hình sentence-transformers
    # Sử dụng mô hình đa ngôn ngữ hỗ trợ tiếng Việt
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Đang tải mô hình: {model_name}")
    model = SentenceTransformer(model_name)
    print("✓ Mô hình đã được tải thành công!\n")
    
    # Bước 2: Load dữ liệu từ CSV
    questions, answers = load_data('data_converted.csv')
    
    if not questions:
        print("✗ Không có dữ liệu để xử lý. Vui lòng kiểm tra file CSV.")
        return
    
    # Bước 3: Encode toàn bộ câu hỏi mẫu thành vector embeddings
    # Việc này chỉ thực hiện một lần khi khởi động để tối ưu hiệu suất
    print("Đang encode các câu hỏi mẫu thành vector embeddings...")
    corpus_embeddings = model.encode(
        questions, 
        convert_to_tensor=True,  # Chuyển sang tensor để tính toán nhanh hơn
        show_progress_bar=True    # Hiển thị thanh tiến trình
    )
    print(f"✓ Đã encode {len(corpus_embeddings)} câu hỏi thành công!\n")
    
    # Bước 4: Vòng lặp tương tác với người dùng
    print("=" * 60)
    print("Chatbot đã sẵn sàng! Gõ 'quit' hoặc 'exit' để thoát.")
    print("=" * 60)
    print()
    
    while True:
        try:
            # Nhận câu hỏi từ người dùng
            user_question = input("Bạn: ").strip()
            
            # Kiểm tra lệnh thoát
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\nCảm ơn bạn đã sử dụng chatbot. Tạm biệt!")
                break
            
            # Bỏ qua câu hỏi rỗng
            if not user_question:
                continue
            
            # Bước 5: Encode câu hỏi của người dùng
            query_embedding = model.encode(
                user_question, 
                convert_to_tensor=True
            )
            
            # Bước 6: Tính độ tương đồng cosine với tất cả câu hỏi mẫu
            # util.cos_sim trả về tensor chứa điểm số similarity cho từng câu hỏi
            cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            
            # Tìm câu hỏi có điểm số cao nhất (top_k=1)
            # argmax() trả về index của phần tử có giá trị lớn nhất
            best_idx = cosine_scores.argmax().item()
            best_score = float(cosine_scores[best_idx])
            
            # Bước 7: Logic trả lời dựa trên độ tin cậy (Confidence-based)
            
            if best_score >= 0.75:
                # Độ tin cậy cao: Trả lời trực tiếp kèm phần trăm
                confidence_percent = best_score * 100
                print(f"\nChatbot: {answers[best_idx]}")
                print(f"         (Độ tin cậy: {confidence_percent:.1f}%)\n")
                
            elif best_score > 0.45:
                # Độ tin cậy trung bình: Hỏi lại để xác nhận
                print(f"\nChatbot: Có phải ý bạn là: \"{questions[best_idx]}\"?")
                print(f"         (Độ tương đồng: {best_score:.2f})\n")
                
            else:
                # Độ tin cậy thấp: Xin lỗi và yêu cầu diễn đạt lại
                print("\nChatbot: Xin lỗi, tôi chưa hiểu ý bạn, vui lòng diễn đạt lại.")
                print(f"         (Độ tương đồng tốt nhất: {best_score:.2f})\n")
        
        except KeyboardInterrupt:
            # Xử lý khi người dùng nhấn Ctrl+C
            print("\n\nCảm ơn bạn đã sử dụng chatbot. Tạm biệt!")
            break
        except Exception as e:
            # Xử lý các lỗi khác
            print(f"\n✗ Lỗi: {e}\n")


if __name__ == "__main__":
    main()

