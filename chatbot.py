#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot hỏi-đáp sử dụng TF-IDF và Cosine Similarity
"""

import csv
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_vietnamese(text):
    """Tiền xử lý tiếng Việt: lowercase, bỏ dấu, chuẩn hóa khoảng trắng"""
    if not isinstance(text, str):
        return text
    # Lowercase
    text = text.lower()
    # Bỏ dấu
    nfd_text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in nfd_text if unicodedata.category(char) != 'Mn')
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class Chatbot:
    """
    Chatbot hỏi-đáp sử dụng TF-IDF và Cosine Similarity
    """
    
    def __init__(self, csv_file='data_converted.csv', similarity_threshold=0.1):
        """
        Khởi tạo chatbot
        
        Args:
            csv_file: Đường dẫn đến file CSV chứa dữ liệu câu hỏi-đáp
            similarity_threshold: Ngưỡng similarity tối thiểu để trả lời (0-1)
        """
        self.csv_file = csv_file
        self.similarity_threshold = similarity_threshold
        self.questions = []
        self.answers = []
        self.processed_questions = []
        self.vectorizer = None
        self.question_vectors = None
        
    def load_data(self):
        """
        Đọc dữ liệu từ file CSV
        """
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    question = row['question'].strip()
                    answer = row['answer'].strip()
                    
                    if question and answer:
                        self.questions.append(question)
                        self.answers.append(answer)
                        # Tiền xử lý câu hỏi để so sánh
                        processed_q = preprocess_vietnamese(question)
                        self.processed_questions.append(processed_q)
            
            print(f"✓ Đã tải {len(self.questions)} cặp câu hỏi-đáp từ {self.csv_file}")
            return True
            
        except FileNotFoundError:
            print(f"✗ Không tìm thấy file: {self.csv_file}")
            return False
        except Exception as e:
            print(f"✗ Lỗi khi đọc file: {e}")
            return False
    
    def train(self):
        """
        Huấn luyện mô hình: vectorize các câu hỏi bằng TF-IDF
        """
        if not self.processed_questions:
            print("✗ Chưa có dữ liệu. Vui lòng load_data() trước.")
            return False
        
        # Khởi tạo TF-IDF Vectorizer
        # Sử dụng ngram_range=(1, 2) để bắt cả từ đơn và cụm 2 từ
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            max_df=0.95
        )
        
        # Vectorize các câu hỏi đã tiền xử lý
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)
        
        print(f"✓ Đã huấn luyện mô hình với {len(self.processed_questions)} câu hỏi")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return True
    
    def find_answer(self, user_question, top_k=1):
        """
        Tìm câu trả lời cho câu hỏi của người dùng
        
        Args:
            user_question: Câu hỏi của người dùng
            top_k: Số lượng câu trả lời tốt nhất cần trả về
            
        Returns:
            Tuple (answer, similarity_score, matched_question) hoặc None nếu không tìm thấy
        """
        if self.vectorizer is None or self.question_vectors is None:
            print("✗ Mô hình chưa được huấn luyện. Vui lòng gọi train() trước.")
            return None
        
        # Tiền xử lý câu hỏi của người dùng
        processed_user_q = preprocess_vietnamese(user_question)
        
        # Vectorize câu hỏi của người dùng
        user_vector = self.vectorizer.transform([processed_user_q])
        
        # Tính cosine similarity với tất cả các câu hỏi trong database
        similarities = cosine_similarity(user_vector, self.question_vectors).flatten()
        
        # Tìm các câu hỏi có similarity cao nhất
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= self.similarity_threshold:
                results.append({
                    'answer': self.answers[idx],
                    'similarity': similarity_score,
                    'matched_question': self.questions[idx]
                })
        
        return results if results else None
    
    def answer(self, user_question, show_details=False):
        """
        Trả lời câu hỏi của người dùng (phương thức chính)
        
        Args:
            user_question: Câu hỏi của người dùng
            show_details: Có hiển thị thông tin chi tiết (similarity, câu hỏi khớp) hay không
            
        Returns:
            Câu trả lời hoặc thông báo không tìm thấy
        """
        results = self.find_answer(user_question, top_k=1)
        
        if results:
            result = results[0]
            answer = result['answer']
            
            if show_details:
                print(f"\n[Similarity: {result['similarity']:.3f}]")
                print(f"[Matched question: {result['matched_question']}]")
            
            return answer
        else:
            return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?"
    
    def interactive_mode(self):
        """
        Chế độ tương tác: cho phép người dùng hỏi liên tục
        """
        print("\n" + "=" * 60)
        print("CHATBOT HỎI-ĐÁP - Chế độ tương tác")
        print("=" * 60)
        print("Gõ 'quit', 'exit' hoặc 'q' để thoát")
        print("Gõ 'help' để xem hướng dẫn")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Thoát
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nCảm ơn bạn đã sử dụng chatbot. Tạm biệt!")
                    break
                
                # Hướng dẫn
                if user_input.lower() == 'help':
                    print("\nHướng dẫn:")
                    print("- Gõ câu hỏi bất kỳ để nhận câu trả lời")
                    print("- Gõ 'quit' để thoát")
                    print("- Gõ 'details' để bật/tắt chế độ hiển thị chi tiết")
                    continue
                
                # Trả lời
                answer = self.answer(user_input, show_details=False)
                print(f"Chatbot: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n\nCảm ơn bạn đã sử dụng chatbot. Tạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}\n")


def main():
    """
    Hàm main để chạy chatbot
    """
    # Khởi tạo chatbot
    chatbot = Chatbot(
        csv_file='data_converted.csv',
        similarity_threshold=0.1  # Có thể điều chỉnh ngưỡng này
    )
    
    # Load dữ liệu
    if not chatbot.load_data():
        return
    
    # Huấn luyện mô hình
    if not chatbot.train():
        return
    
    # Test với một số câu hỏi mẫu
    print("\n" + "=" * 60)
    print("TEST CHATBOT")
    print("=" * 60)
    
    test_questions = [
        "AI là gì?",
        "Chatbot hoạt động như thế nào?",
        "Trí tuệ nhân tạo được ứng dụng ở đâu?",
        "Làm sao để xây dựng chatbot?",
        "Xin chào"  # Câu hỏi không có trong database
    ]
    
    for question in test_questions:
        print(f"\nCâu hỏi: {question}")
        answer = chatbot.answer(question, show_details=True)
        print(f"Trả lời: {answer}")
    
    # Chạy chế độ tương tác
    print("\n" + "=" * 60)
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()

