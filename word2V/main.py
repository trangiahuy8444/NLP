import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Tập dữ liệu: Các mô tả sản phẩm trên hệ thống
documents = [
    "Điện thoại thông minh camera sắc nét pin trâu",
    "Smartphone chụp ảnh đẹp thời lượng pin dài",
    "Laptop văn phòng mỏng nhẹ hiệu năng cao",
    "Máy tính xách tay cấu hình mạnh cho dân công sở",
    "Tai nghe không dây chống ồn chủ động âm bass tốt",
    "Headphone bluetooth pin khủng cách âm tốt"
]

def preprocess(text):
    # Đưa về chữ thường và tách từ theo khoảng trắng
    return text.lower().split()

# Áp dụng cho toàn bộ tập dữ liệu
tokenized_docs = [preprocess(doc) for doc in documents]
print("Dữ liệu sau khi xử lý:", tokenized_docs[0]) 
# Output: ['điện', 'thoại', 'thông', 'minh', 'camera', 'sắc', 'nét', 'pin', 'trâu']

# vector_size: Kích thước của vector từ (thường là 100-300, ở đây dữ liệu ít nên dùng 50)
# window: Số lượng từ xung quanh được xét đến
# min_count: Bỏ qua những từ xuất hiện ít hơn 1 lần
model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1, workers=4)

# print("Vector của từ 'pin':\n", model.wv['pin'][:5]) # In 5 chiều đầu tiên

def get_document_vector(doc_tokens, model):
    # Lấy vector của các từ có trong từ điển của mô hình
    vectors = [model.wv[word] for word in doc_tokens if word in model.wv]
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
        
    # Tính trung bình cộng theo chiều dọc (axis=0)
    return np.mean(vectors, axis=0)

# Trích xuất vector cho toàn bộ sản phẩm trong database
doc_vectors = [get_document_vector(tokens, model) for tokens in tokenized_docs]

def search_similar_products(query, top_n=2):
    # 1. Tiền xử lý câu truy vấn
    query_tokens = preprocess(query)
    
    # 2. Biến câu truy vấn thành vector
    query_vec = get_document_vector(query_tokens, model).reshape(1, -1)
    db_vecs = np.array(doc_vectors)
    
    # 3. Tính độ tương đồng toán học
    similarities = cosine_similarity(query_vec, db_vecs)[0]
    
    # 4. Lấy ra top_n sản phẩm có điểm cao nhất
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    print(f"--- Kết quả cho truy vấn: '{query}' ---")
    for i in top_indices:
        print(f"Khớp {similarities[i]*100:.1f}% : {documents[i]}")

# Thử nghiệm sản phẩm thực tế!
search_similar_products("tìm smartphone có pin lâu")
search_similar_products("cần mua máy tính văn phòng")