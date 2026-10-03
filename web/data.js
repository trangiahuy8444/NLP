// Dữ liệu tri thức và mô phỏng cho Web Interactive Báo Cáo Giữa Kỳ NLP
// Giảng viên hướng dẫn: PGS.TS. LÊ ANH CƯỜNG - ĐH Sư Phạm TP.HCM
// Nhóm sinh viên: Trần Gia Huy (KHMT836012), Đỗ Minh Khánh Ngân (KHMT836019), Nguyễn Tấn Phát (KHMT836026)

const NLP_DATA = {
  projectInfo: {
    title: "MÔ HÌNH WORD2VEC, WORD EMBEDDING VÀ HỌC BIỂU DIỄN VĂN BẢN BẰNG LSTM CHO BÀI TOÁN PHÂN LOẠI CẢM XÚC",
    subtitle: "Nghiên cứu đối chuẩn 6 mô hình trên Đa tập dữ liệu Thực tế (UIT-VSFC & E-Commerce), nâng cấp BiLSTM-Attention và PhoBERT SOTA",
    institution: "Trường Đại học Sư phạm Thành phố Hồ Chí Minh - Khoa Khoa học Máy tính",
    course: "Xử lý ngôn ngữ tự nhiên (Natural Language Processing) - Khóa 36 (2025-2027)",
    instructor: "PGS.TS. LÊ ANH CƯỜNG",
    students: [
      { name: "Trần Gia Huy", id: "KHMT836012", role: "Trưởng nhóm - BiLSTM & Self-Attention, Báo cáo & Thuyết trình" },
      { name: "Đỗ Minh Khánh Ngân", id: "KHMT836019", role: "Thành viên - Tiền xử lý dữ liệu, Huấn luyện Word2Vec & Đối chuẩn TF-IDF" },
      { name: "Nguyễn Tấn Phát", id: "KHMT836026", role: "Thành viên - Fine-tuning PhoBERT SOTA, Trích xuất t-SNE & Confusion Matrix" }
    ]
  },

  // 1. Dữ liệu từ vựng & Không gian Vector mô phỏng
  vectorPlayground: {
    words: [
      { word: "thầy_cô", vec2d: [0.78, 0.62], dense: [0.78, 0.62, 0.45, 0.88, 0.12], onehotIdx: 42, category: "Giáo dục" },
      { word: "giảng_viên", vec2d: [0.82, 0.58], dense: [0.82, 0.58, 0.48, 0.85, 0.15], onehotIdx: 108, category: "Giáo dục" },
      { word: "nhiệt_tình", vec2d: [0.65, 0.75], dense: [0.65, 0.75, 0.82, 0.90, 0.05], onehotIdx: 215, category: "Tích cực" },
      { word: "tận_tâm", vec2d: [0.69, 0.72], dense: [0.69, 0.72, 0.80, 0.89, 0.08], onehotIdx: 340, category: "Tích cực" },
      { word: "xuất_sắc", vec2d: [0.58, 0.82], dense: [0.58, 0.82, 0.91, 0.78, 0.04], onehotIdx: 512, category: "Tích cực" },
      { word: "thất_vọng", vec2d: [-0.72, -0.65], dense: [-0.72, -0.65, -0.85, -0.70, 0.90], onehotIdx: 680, category: "Tiêu cực" },
      { word: "rất_kém", vec2d: [-0.79, -0.58], dense: [-0.79, -0.58, -0.88, -0.75, 0.85], onehotIdx: 792, category: "Tiêu cực" },
      { word: "chán_nản", vec2d: [-0.68, -0.72], dense: [-0.68, -0.72, -0.82, -0.68, 0.88], onehotIdx: 855, category: "Tiêu cực" },
      { word: "điện_thoại", vec2d: [0.15, -0.75], dense: [0.15, -0.75, 0.05, -0.10, 0.35], onehotIdx: 1200, category: "Công nghệ" },
      { word: "smartphone", vec2d: [0.18, -0.72], dense: [0.18, -0.72, 0.08, -0.08, 0.32], onehotIdx: 1450, category: "Công nghệ" },
      { word: "giao_hàng", vec2d: [-0.25, -0.55], dense: [-0.25, -0.55, 0.12, -0.30, 0.60], onehotIdx: 1820, category: "Thương mại" },
      { word: "chất_lượng", vec2d: [0.35, 0.40], dense: [0.35, 0.40, 0.50, 0.42, 0.20], onehotIdx: 2100, category: "Thương mại" }
    ],
    vocabSize: 10000
  },

  // 2. Dữ liệu mô phỏng Word2Vec Sliding Window
  word2vecSentences: [
    {
      id: "s1",
      domain: "UIT-VSFC",
      tokens: ["giáo_viên", "rất", "nhiệt_tình", "giảng_bài", "dễ_hiểu", "cho", "sinh_viên"],
      defaultCenter: 2 // "nhiệt_tình"
    },
    {
      id: "s2",
      domain: "UIT-VSFC",
      tokens: ["phương_pháp", "dạy", "còn", "khô_khan", "gây", "buồn_ngủ", "trên", "lớp"],
      defaultCenter: 3 // "khô_khan"
    },
    {
      id: "s3",
      domain: "E-Commerce",
      tokens: ["sản_phẩm", "đẹp", "tuyệt_vời", "đóng_gói", "cẩn_thận", "giao_hàng", "nhanh"],
      defaultCenter: 2 // "tuyệt_vời"
    },
    {
      id: "s4",
      domain: "E-Commerce",
      tokens: ["thất_vọng", "chất_lượng", "quá", "tệ", "hàng", "bị", "rách_nát"],
      defaultCenter: 3 // "tệ"
    }
  ],

  // 3. Mô phỏng cạm bẫy túi từ (Average Word2Vec Flaw)
  avgW2vFlawSamples: [
    {
      title: "Trường hợp 1: Đảo vị trí từ nhưng giữ nguyên từ vựng",
      sentA: "phim này hay chứ không dở",
      sentB: "phim này dở chứ không hay",
      meaningA: "Khen phim (Tích cực: Hay)",
      meaningB: "Chê phim (Tiêu cực: Dở)",
      tokensA: ["phim", "này", "hay", "chứ", "không", "dở"],
      tokensB: ["phim", "này", "dở", "chứ", "không", "hay"],
      mathExplanation: "Do phép toán của Average Word2Vec là v_D = 1/N * sum(v_w), tính giao hoán của phép cộng làm cho Vector(Câu A) = Vector(Câu B) 100%! Mô hình Logistic Regression hoàn toàn bất lực không thể phân biệt được đâu là khen, đâu là chê."
    },
    {
      title: "Trường hợp 2: Phủ định kép và chuyển hướng cảm xúc",
      sentA: "thầy giảng không phải là không tận tâm",
      sentB: "thầy giảng tận tâm nhưng không dễ hiểu",
      meaningA: "Khẳng định ngầm là thầy tận tâm (Tích cực)",
      meaningB: "Khen nhưng vế sau chê (Tiêu cực / Phàn nàn)",
      tokensA: ["thầy", "giảng", "không", "phải", "là", "không", "tận_tâm"],
      tokensB: ["thầy", "giảng", "tận_tâm", "nhưng", "không", "dễ_hiểu"],
      mathExplanation: "Average Word2Vec chỉ đơn giản gom nhặt từ khóa rời rạc, làm triệt tiêu hoàn toàn liên kết cấu trúc cú pháp tuần tự mà chỉ mạng hồi quy như LSTM mới mô hình hóa được."
    }
  ],

  // 4. Dữ liệu mô phỏng từng bước 3 Cổng LSTM (LSTM Gate Stepper)
  lstmSimulation: {
    sentence: ["thầy", "giảng", "rất", "nhiệt_tình", "nhưng", "đề", "thi", "quá", "khó"],
    steps: [
      {
        token: "thầy",
        pos: 1,
        forgetGate: 0.95, // Giữ lại ngữ cảnh ban đầu
        inputGate: 0.60,  // Nạp thực thể chủ ngữ
        candidateVal: 0.10, // Trung tính
        cellStateVal: 0.10, // Bắt đầu tích lũy
        outputGate: 0.50,
        hiddenStateVal: 0.05,
        note: "Khởi tạo ngữ cảnh: Đang nói về chủ thể 'thầy' trong môi trường giáo dục."
      },
      {
        token: "giảng",
        pos: 2,
        forgetGate: 0.90,
        inputGate: 0.70,
        candidateVal: 0.20,
        cellStateVal: 0.23,
        outputGate: 0.55,
        hiddenStateVal: 0.12,
        note: "Hành động giảng dạy của người thầy."
      },
      {
        token: "rất",
        pos: 3,
        forgetGate: 0.92,
        inputGate: 0.85, // Từ chỉ mức độ cao
        candidateVal: 0.40,
        cellStateVal: 0.55,
        outputGate: 0.65,
        hiddenStateVal: 0.32,
        note: "Khuếch đại mức độ cảm xúc chuẩn bị theo sau."
      },
      {
        token: "nhiệt_tình",
        pos: 4,
        forgetGate: 0.88,
        inputGate: 0.95, // Trọng số cảm xúc cực lớn
        candidateVal: 0.90, // Tích cực mạnh
        cellStateVal: 0.92, // Đỉnh cao tích cực
        outputGate: 0.85,
        hiddenStateVal: 0.82,
        note: "Cực đại cảm xúc tích cực! Cell state ghi nhận trạng thái khen ngợi mạnh mẽ."
      },
      {
        token: "nhưng",
        pos: 5,
        forgetGate: 0.25, // BƯỚC NGOẶT: CỔNG QUÊN KÍCH HOẠT MẠNH (XÓA KÝ ỨC CŨ)
        inputGate: 0.75, // Chuẩn bị nạp thông tin đảo chiều
        candidateVal: -0.30,
        cellStateVal: 0.15, // Cell state bị sụt giảm từ 0.92 xuống 0.15
        outputGate: 0.45,
        hiddenStateVal: 0.07,
        note: "⚠️ BƯỚC NGOẶT ĐẢO CHIỀU: Gặp liên từ 'nhưng', Forget Gate hạ xuống 0.25 để TẨY XÓA bớt ký ức tích cực trước đó, chuẩn bị tiếp nhận ý đối lập."
      },
      {
        token: "đề",
        pos: 6,
        forgetGate: 0.80,
        inputGate: 0.60,
        candidateVal: -0.20,
        cellStateVal: 0.02,
        outputGate: 0.50,
        hiddenStateVal: 0.01,
        note: "Chuyển đối tượng sang bài thi, kiểm tra."
      },
      {
        token: "thi",
        pos: 7,
        forgetGate: 0.85,
        inputGate: 0.65,
        candidateVal: -0.25,
        cellStateVal: -0.14,
        outputGate: 0.55,
        hiddenStateVal: -0.07,
        note: "Ngữ cảnh kỳ thi căng thẳng."
      },
      {
        token: "quá",
        pos: 8,
        forgetGate: 0.90,
        inputGate: 0.85,
        candidateVal: -0.50,
        cellStateVal: -0.52,
        outputGate: 0.75,
        hiddenStateVal: -0.36,
        note: "Từ bổ trợ chỉ mức độ tiêu cực tăng cao."
      },
      {
        token: "khó",
        pos: 9,
        forgetGate: 0.92,
        inputGate: 0.95,
        candidateVal: -0.88, // Tiêu cực mạnh về độ khó
        cellStateVal: -0.85, // Cell state đảo sang âm
        outputGate: 0.90,
        hiddenStateVal: -0.78, // Vector biểu diễn cuối cùng phản ánh cảm xúc than phiền/tiêu cực
        note: "KẾT LUẬN CỦA LSTM: Trạng thái ẩn cuối cùng h_T phản ánh chính xác cấu trúc chuyển ngữ nghĩa. Cảm xúc chốt lại là than phiền (Tiêu cực: 0)."
      }
    ]
  },

  // 5. Dữ liệu Attention Heatmap thực nghiệm từ mô hình thật
  attentionSamples: [
    {
      id: "uit_pos",
      domain: "UIT-VSFC (Giáo dục)",
      type: "Tích cực (Nhãn 1)",
      modelPred: "Tích cực (98.4%)",
      tokens: [
        { word: "giáo_viên", weight: 0.08 },
        { word: "nhiệt_tình", weight: 0.42 },
        { word: "gần_gũi", weight: 0.35 },
        { word: "với", weight: 0.05 },
        { word: "sinh_viên", weight: 0.07 },
        { word: ".", weight: 0.03 }
      ],
      comment: "Mô hình Self-Attention tập trung tới 77% trọng số vào hai từ khóa cảm xúc cốt lõi: 'nhiệt_tình' (0.42) và 'gần_gũi' (0.35). Các hư từ như 'với', dấu chấm chỉ nhận trọng số dưới 0.05.",
      imagePath: "results/uit_vsfc/attention_sample_pos.png"
    },
    {
      id: "uit_neg",
      domain: "UIT-VSFC (Giáo dục)",
      type: "Tiêu cực (Nhãn 0)",
      modelPred: "Tiêu cực (96.8%)",
      tokens: [
        { word: "phương_pháp", weight: 0.06 },
        { word: "dạy", weight: 0.07 },
        { word: "không", weight: 0.28 },
        { word: "được", weight: 0.05 },
        { word: "thu_hút", weight: 0.38 },
        { word: "lắm", weight: 0.12 },
        { word: ".", weight: 0.04 }
      ],
      comment: "Từ 'không' (0.28) và 'thu_hút' (0.38) kết hợp lại tạo thành cụm phủ định được gán trọng số áp đảo (66% tổng sự chú ý), giúp mô hình bắt đúng sắc thái chê bai.",
      imagePath: "results/uit_vsfc/attention_sample_neg.png"
    },
    {
      id: "ecom_pos",
      domain: "E-Commerce (Thương mại)",
      type: "Tích cực (Nhãn 1)",
      modelPred: "Tích cực (99.1%)",
      tokens: [
        { word: "hàng", weight: 0.04 },
        { word: "đẹp", weight: 0.32 },
        { word: "tuyệt_vời", weight: 0.41 },
        { word: "giao_hàng", weight: 0.06 },
        { word: "siêu", weight: 0.05 },
        { word: "nhanh", weight: 0.12 }
      ],
      comment: "Trong miền thương mại, các tính từ mạnh 'đẹp' (0.32) và 'tuyệt_vời' (0.41) giữ vai trò quyết định phân loại sắc thái tích cực.",
      imagePath: "results/ecommerce/attention_sample_pos.png"
    },
    {
      id: "ecom_neg",
      domain: "E-Commerce (Thương mại)",
      type: "Tiêu cực (Nhãn 0)",
      modelPred: "Tiêu cực (97.5%)",
      tokens: [
        { word: "thất_vọng", weight: 0.45 },
        { word: "chất_lượng", weight: 0.08 },
        { word: "sản_phẩm", weight: 0.05 },
        { word: "rất", weight: 0.09 },
        { word: "kém", weight: 0.33 }
      ],
      comment: "Từ đầu câu 'thất_vọng' đã chiếm tới 45% trọng số, kết hợp với 'kém' (33%) khiến mô hình đưa ra dự đoán nhãn Tiêu cực gần như tuyệt đối.",
      imagePath: "results/ecommerce/attention_sample_neg.png"
    }
  ],

  // 6. Bảng tổng hợp đối chuẩn 6 mô hình trên Đa tập dữ liệu
  benchmarkResults: [
    {
      id: 1,
      model: "TF-IDF + Logistic Regression (Baseline 1)",
      category: "Cổ điển (Bag-of-Words)",
      uitAcc: 89.83,
      uitF1: 89.80,
      ecomAcc: 70.33,
      ecomF1: 69.65,
      trainTime: "0.8s",
      vectorDim: "~4,500",
      pros: "Tính toán cực nhanh, đơn giản, dễ triển khai",
      cons: "Mất hoàn toàn thứ tự từ, không gian thưa, không hiểu đồng nghĩa",
      cmUit: "results/uit_vsfc/cm_tfidf.png",
      cmEcom: "results/ecommerce/cm_tfidf.png"
    },
    {
      id: 2,
      model: "Average Word2Vec + Logistic Regression (Baseline 2)",
      category: "Vector Phân bố (Thống kê)",
      uitAcc: 87.50,
      uitF1: 87.49,
      ecomAcc: 67.33,
      ecomF1: 66.80,
      trainTime: "1.2s",
      vectorDim: "100",
      pros: "Kích thước vector dày đặc cố định d=100, nắm bắt ngữ nghĩa từ",
      cons: "Phép trung bình cộng xóa nhòa cú pháp, đảo ngữ và phủ định kép",
      cmUit: "results/uit_vsfc/cm_avg_word2vec.png",
      cmEcom: "results/ecommerce/cm_avg_word2vec.png"
    },
    {
      id: 3,
      model: "BiLSTM Classifier (Học từ đầu - Scratch)",
      category: "Học sâu Hồi quy (Deep Learning)",
      uitAcc: 89.17,
      uitF1: 89.16,
      ecomAcc: 69.17,
      ecomF1: 68.75,
      trainTime: "45s",
      vectorDim: "256 (h_T 2 chiều)",
      pros: "Mô hình hóa chuỗi 2 chiều, Cell State ghi nhớ dài hạn",
      cons: "Dễ quá khớp (overfitting) khi lượng dữ liệu huấn luyện còn khiêm tốn",
      cmUit: "results/uit_vsfc/cm_bilstm_scratch.png",
      cmEcom: "results/ecommerce/cm_bilstm_scratch.png"
    },
    {
      id: 4,
      model: "BiLSTM Classifier (Pre-trained Word2Vec)",
      category: "Học sâu Kết hợp Pre-trained",
      uitAcc: 91.00,
      uitF1: 91.00,
      ecomAcc: 67.17,
      ecomF1: 67.05,
      trainTime: "42s",
      vectorDim: "256",
      pros: "Hưởng lợi từ không gian vector Word2Vec huấn luyện trên kho ngữ liệu",
      cons: "Từ ngoài từ điển (OOV), tiếng lóng TMĐT chưa có trong Word2Vec",
      cmUit: "results/uit_vsfc/cm_bilstm_pretrained.png",
      cmEcom: "results/ecommerce/cm_bilstm_pretrained.png"
    },
    {
      id: 5,
      model: "BiLSTM + Self-Attention (Cơ chế Đề xuất)",
      category: "Học sâu Chú ý Thích nghi",
      uitAcc: 91.83,
      uitF1: 91.83,
      ecomAcc: 72.50,
      ecomF1: 72.40,
      trainTime: "52s",
      vectorDim: "128 (Tổng hợp có trọng số)",
      pros: "Bứt phá +5.33% trên TMĐT, trực quan hóa trọng số Attention Heatmap",
      cons: "Vẫn bị hạn chế bởi kiến trúc tuần tự O(N) khi câu quá dài",
      cmUit: "results/uit_vsfc/cm_bilstm_attention.png",
      cmEcom: "results/ecommerce/cm_bilstm_attention.png"
    },
    {
      id: 6,
      model: "PhoBERT-base-v2 (Transformer Tiếng Việt SOTA)",
      category: "Mô hình Ngôn ngữ Lớn Tiếng Việt",
      uitAcc: 95.50,
      uitF1: 95.50,
      ecomAcc: 86.17,
      ecomF1: 86.16,
      trainTime: "3m 40s (MPS GPU)",
      vectorDim: "768 ([CLS] token)",
      pros: "Áp đảo tuyệt đối (+15.84% TMĐT), hiểu sâu sắc đa nghĩa và teencode",
      cons: "Kích thước mô hình lớn (540MB), đòi hỏi phần cứng GPU huấn luyện",
      cmUit: "results/uit_vsfc/cm_phobert.png",
      cmEcom: "results/ecommerce/cm_phobert.png"
    }
  ],

  // 7. Ngân hàng câu hỏi Vấn đáp bảo vệ (Mock Defense with PGS.TS. LÊ ANH CƯỜNG)
  defenseQuestions: [
    {
      id: "q1",
      topic: "Word Embedding & Word2Vec",
      question: "Câu hỏi 1: Vì sao One-Hot Encoding lại gặp 'lời nguyền số chiều' và tính trực giao? Word2Vec khắc phục điều đó như thế nào?",
      intuitive: "Nói đơn giản: One-Hot giống như cấp cho mỗi từ 1 mã số riêng trong danh bạ 10.000 số. Hai từ 'thầy' và 'cô' dù nghĩa rất gần nhau nhưng mã nhị phân chả liên quan gì nhau (tích vô hướng = 0). Word2Vec thì cho mỗi từ một 'tọa độ địa lý' 100 số thực, từ nào đồng nghĩa thì tọa độ nằm sát bên cạnh nhau, máy tính chỉ cần đo khoảng cách là biết ngay!",
      academic: "One-Hot biểu diễn từ w_i thành vector e_i trong {0, 1}^|V|. Hai nhược điểm chí mạng là: (1) Chiều không gian bùng nổ theo kích thước từ điển |V|, gây ma trận cực thưa (sparse); (2) Các vector hoàn toàn trực giao: e_i^T * e_j = 0 với mọi i != j, không phản ánh được quan hệ ngữ nghĩa. Word2Vec dựa trên Giả thuyết phân bố (Distributional Hypothesis - Firth 1957) ánh xạ từ vào không gian dày đặc R^d (d=100 << |V|), học trọng số để cực đại hóa xác suất đồng xuất hiện.",
      gotcha: "Lưu ý câu gài của thầy: 'Có phải cứ tăng số chiều vector d lên 1000 hay 5000 là Word2Vec sẽ thông minh hơn không?'. Trả lời: Không, số chiều quá lớn dẫn đến lãng phí bộ nhớ và overfitting; với tập ngữ liệu vừa và nhỏ, d=100 đến 300 là điểm cân bằng tối ưu."
    },
    {
      id: "q2",
      topic: "Word2Vec (CBOW vs Skip-Gram)",
      question: "Câu hỏi 2: Phân biệt kiến trúc CBOW và Skip-Gram. Trong trường hợp nào thì Skip-Gram vượt trội hơn CBOW?",
      intuitive: "CBOW giống như trò chơi 'Điền vào chỗ trống': Cho 4 từ xung quanh, đoán từ ở giữa. Skip-Gram thì ngược lại: Cho 1 từ ở giữa, bắt máy đoán 4 từ xung quanh. Skip-Gram giỏi hơn CBOW khi gặp các từ hiếm (từ ít xuất hiện), vì mỗi lần từ hiếm xuất hiện nó được cập nhật trực tiếp với từng từ ngữ cảnh!",
      academic: "CBOW dự đoán từ đích w_t dựa trên trung bình cộng vector của 2c từ ngữ cảnh xung quanh; hàm mục tiêu L_CBOW = sum log P(w_t | w_{t-c}, ..., w_{t+c}). Skip-Gram dùng từ trung tâm w_t để dự đoán lần lượt từng từ ngữ cảnh; L_SG = sum sum log P(w_{t+j} | w_t). CBOW huấn luyện nhanh gấp vài lần và mượt mà trên từ phổ biến; Skip-Gram nắm bắt rất tốt các từ hiếm và ngữ nghĩa đa dạng vì không bị hiệu ứng cào bằng (averaging effect).",
      gotcha: "Thầy có thể hỏi độ phức tạp: Softmax nguyên bản tốn O(|V|) tại mỗi bước. Vì vậy Mikolov đã đề xuất Negative Sampling để giảm xuống O(K) với K=5 đến 20 mẫu âm."
    },
    {
      id: "q3",
      topic: "Negative Sampling (SGNS)",
      question: "Câu hỏi 3: Negative Sampling biến đổi bài toán như thế nào? Tại sao lại lấy lũy thừa 3/4 trong hàm phân phối Unigram?",
      intuitive: "Thay vì bắt máy tính thi trắc nghiệm 50.000 đáp án (mẫu số Softmax cực nặng), Negative Sampling đổi thành bài kiểm tra Đúng/Sai nhị phân: Cặp này có đi chung không? Đúng (mẫu dương) hay Sai (5 mẫu âm ngẫu nhiên). Mũ 3/4 là 'thuật bù trừ': làm cho các từ hiếm gặp có thêm cơ hội được chọn làm mẫu âm, tránh việc chỉ toàn bốc phải từ 'và', 'thì', 'là'.",
      academic: "SGNS chuyển đổi bài toán đa lớp sang phân loại nhị phân Logistic Regression: log sigma(v'_wO^T * v_wI) + sum_k E[log sigma(-v'_wnk^T * v_wI)]. Mẫu âm được rút từ phân phối Unigram mũ 3/4: P_n(w) = (f(w)^{3/4}) / (sum f(w')^{3/4}). Số mũ 0.75 làm giảm nhẹ xác suất của các từ siêu phổ biến (stop words) và tăng xác suất xuất hiện của các từ tần suất thấp, giúp vector của từ hiếm được cập nhật cân bằng hơn.",
      gotcha: "Công thức toán: Hãy nhớ hàm sigmoid sigma(x) = 1 / (1 + exp(-x)) và lý do dùng mẫu âm thay vì toàn bộ từ điển."
    },
    {
      id: "q4",
      topic: "Biểu diễn văn bản (Document Representation)",
      question: "Câu hỏi 4: Tại sao Average Word2Vec lại thất bại trước các câu đảo ngữ như 'phim này hay chứ không dở' và 'phim này dở chứ không hay'?",
      intuitive: "Vì phép cộng có tính chất giao hoán (2+3 cũng bằng 3+2)! Khi ta cộng dồn tất cả các vector từ lại rồi chia cho số lượng từ, thứ tự trước sau bị mất sạch sành sanh. Cả hai câu trên đều chứa đúng 6 từ đó, nên vector trung bình của chúng giống hệt nhau 100%, máy hoàn toàn mù tịt không biết câu nào khen câu nào chê!",
      academic: "Average Word2Vec định nghĩa v_D = 1/|D| * sum_{w in D} v_w. Đây là mô hình túi từ liên tục (Continuous Bag-of-Words), hoàn toàn bỏ qua cấu trúc ngữ pháp, phụ thuộc tuần tự thời gian và ngữ cảnh cục bộ. Do đó, hiện tượng phủ định kép, từ bổ ngữ đảo vị trí bị triệt tiêu hoàn toàn.",
      gotcha: "Câu hỏi nối tiếp của thầy: 'Vậy có cách nào cải tiến Average Word2Vec mà không cần dùng LSTM không?'. Trả lời: Có thể dùng TF-IDF Weighted Word2Vec (gán trọng số theo độ quan trọng của từ) hoặc mô hình Doc2Vec (PV-DM / PV-DBOW)."
    },
    {
      id: "q5",
      topic: "Kiến trúc Mạng LSTM",
      question: "Câu hỏi 5: Cell State (Trạng thái ô nhớ) và Hidden State (Trạng thái ẩn) trong LSTM khác nhau như thế nào? Vì sao LSTM giải quyết được Vanishing Gradient?",
      intuitive: "Cell State c_t giống như 'đường cao tốc băng chuyền' chạy xuyên suốt thời gian, trên đó thông tin chỉ bị cộng thêm hoặc nhân với cổng quên, không bị bóp méo liên tục qua hàm kích hoạt phi tuyến. Hidden State h_t là 'bản tóm tắt tức thời' được lọc ra từ Cell State qua Output Gate để chuẩn bị xuất ra quyết định.",
      academic: "Trong RNN truyền thống, đạo hàm lan truyền ngược bị nhân dồn dập qua ma trận trọng số W và hàm tanh/sigmoid, dẫn tới đạo hàm tiến về 0 (Vanishing Gradient). Trong LSTM, Cell State cập nhật tuyến tính: c_t = f_t * c_{t-1} + i_t * c_tilde_t. Đạo hàm d(c_t)/d(c_{t-1}) chứa số hạng f_t trực tiếp. Nếu cổng quên mở (f_t gần 1), gradient có thể truyền ngược qua hàng chục bước thời gian mà không bị triệt tiêu.",
      gotcha: "Thầy sẽ hỏi: '3 Cổng của LSTM dùng hàm kích hoạt gì và vì sao?'. Trả lời: 3 Cổng (f_t, i_t, o_t) dùng hàm Sigmoid vì giá trị nằm trong đoạn [0, 1] đại diện cho tỷ lệ đóng/mở cổng (0% là chặn hoàn toàn, 100% là cho qua hết)."
    },
    {
      id: "q6",
      topic: "Mạng BiLSTM (Hai chiều)",
      question: "Câu hỏi 6: Trong đồ án, vector biểu diễn toàn văn bản v_D được trích xuất từ BiLSTM bằng công thức nào? Vì sao lại dùng BiLSTM thay vì LSTM một chiều?",
      intuitive: "Nếu đọc từ trái sang phải, ta chỉ hiểu được ngữ cảnh của các từ đứng trước. Nhưng trong tiếng Việt, từ đứng sau thường quyết định nghĩa của từ đứng trước (ví dụ: 'thực sự rất... tệ'). BiLSTM đọc câu theo cả 2 hướng: từ đầu đến cuối và từ cuối về đầu, rồi ghép 2 đầu lại với nhau thành 1 vector 256 chiều gom trọn ý nghĩa cả câu!",
      academic: "BiLSTM chạy song song 2 chuỗi hồi quy: chuỗi xuôi (forward) h_t_right và chuỗi ngược (backward) h_t_left. Vector văn bản v_D được tạo bằng cách ghép nối trạng thái ẩn bước cuối cùng của cả 2 chiều: v_D = [h_T_right ; h_1_left] in R^{2 * d_hidden} = R^{256}. Nhờ đó, vector văn bản nắm bắt đồng thời mối liên hệ xuôi dòng và ngược dòng của toàn bộ văn bản.",
      gotcha: "Lưu ý phương pháp pooling: Nhóm đã thử nghiệm cả 'last' (lấy bước cuối) và 'mean-pooling' (trung bình trạng thái ẩn có che mặt nạ padding). Phương pháp nối bước cuối cho kết quả tối ưu nhất trên bài toán phân loại nhị phân này."
    },
    {
      id: "q7",
      topic: "Cơ chế Self-Attention",
      question: "Câu hỏi 7: Cơ chế Self-Attention hoạt động ra sao trong mô hình BiLSTM-Attention của nhóm? Nó mang lại lợi ích gì về tính khả giải thích?",
      intuitive: "Bình thường BiLSTM cố nén cả câu dài vào 1 vector cuối cùng, rất dễ bị nghẽn thông tin. Self-Attention giống như người đọc cầm bút dạ quang: lướt qua cả câu, từ nào quan trọng (như 'tuyệt_vời', 'thất_vọng') thì tô màu đậm (trọng số cao 0.4), từ nào râu ria ('là', 'với') thì tô nhạt (0.03). Nhờ vậy nhóm xuất ra được bản đồ nhiệt Attention Heatmap giải thích rõ ràng vì sao máy đoán nhãn đó!",
      academic: "Mô hình dùng Additive Attention (Bahdanau): u_t = tanh(W_a * h_t + b_a), score e_t = u_t^T * v_a, trọng số alpha_t = softmax(e_t). Vector văn bản là tổ hợp tuyến tính v_D = sum_t (alpha_t * h_t). Lợi ích: (1) Tránh điểm nghẽn nén ép thông tin của LSTM; (2) Tăng tính khả giải thích (Interpretability) thông qua phân phối trọng số alpha_t trực quan hóa thành Attention Heatmaps.",
      gotcha: "Thực nghiệm chứng minh: Trên tập E-Commerce, BiLSTM-Attention tăng vọt từ 67.17% lên 72.50% (+5.33%), chứng tỏ câu càng lộn xộn thì cơ chế Attention càng phát huy sức mạnh!"
    },
    {
      id: "q8",
      topic: "Đa miền dữ liệu & Sự sụt giảm trên E-Commerce",
      question: "Câu hỏi 8: Tại sao tất cả các mô hình khi chuyển từ tập UIT-VSFC sang E-Commerce đều bị sụt giảm độ chính xác từ 10% đến 20%?",
      intuitive: "Vì học sinh viết phản hồi cho thầy cô (UIT-VSFC) dùng từ ngữ lịch sự, câu cú đàng hoàng, ngữ pháp chuẩn. Còn người mua hàng trên mạng (E-Commerce) thì dùng từ lóng, teencode ('k', 'ko', 'đc', 'ship'), viết tắt, thả icon, châm biếm ('cho 5 sao để hiện lên đầu nhưng hàng như hạch'). Ngôn ngữ mạng xã hội cực kỳ nhiễu và phức tạp hơn rất nhiều!",
      academic: "Sự suy giảm hiệu năng là hệ quả của hiện tượng dịch chuyển miền ngôn ngữ (Domain Shift) và nhiễu từ vựng: (1) Tỷ lệ từ ngoài từ điển (Out-Of-Vocabulary - OOV) trên TMĐT cao gấp 3.5 lần do teencode, lỗi chính tả cố ý; (2) Cấu trúc câu tự do, đảo trật tự ngữ pháp; (3) Tần suất xuất hiện sắc thái châm biếm (sarcasm/irony) cao, làm các mô hình n-gram và LSTM thuần túy phân loại sai.",
      gotcha: "Dẫn chứng số liệu thực tế đồ án: TF-IDF giảm từ 89.83% xuống 70.33% (-19.50%), BiLSTM giảm từ 91.00% xuống 67.17% (-23.83%). Duy nhất PhoBERT duy trì được độ chính xác ấn tượng 86.17%!"
    },
    {
      id: "q9",
      topic: "Mô hình PhoBERT Transformer SOTA",
      question: "Câu hỏi 9: Vì sao PhoBERT lại vượt trội áp đảo (+15.84% so với TF-IDF trên tập E-Commerce)? Khác biệt cốt lõi giữa Word2Vec và PhoBERT là gì?",
      intuitive: "Word2Vec là biểu diễn 'tĩnh': Từ 'cơm' trong 'ăn cơm' hay 'cơm áo gạo tiền' đều có đúng 1 vector như nhau. PhoBERT là biểu diễn 'động theo ngữ cảnh': tùy câu văn mà từ đó sẽ có vector biến đổi linh hoạt! Thêm nữa, PhoBERT đã được học sẵn trên 20GB báo chí và văn bản tiếng Việt (~3 tỷ từ), có 12 tầng Transformer phân tích ngữ nghĩa sâu sắc nên đọc đâu hiểu đó!",
      academic: "Khác biệt cốt lõi: Word2Vec là Static Word Embedding (mỗi từ có 1 vector cố định trong bảng tra cứu), không giải quyết được từ đồng âm khác nghĩa. PhoBERT là Contextual Language Model dựa trên kiến trúc RoBERTa (Vaswani et al. 2017), tiền huấn luyện trên 20GB ngữ liệu tiếng Việt qua nhiệm vụ Masked Language Modeling (MLM). Cơ chế 12 tầng Multi-Head Self-Attention tính toán sự tương tác giữa mọi cặp từ ở mọi tầng, trích xuất đặc trưng ngữ nghĩa sâu sắc ở token [CLS].",
      gotcha: "Thầy sẽ hỏi: 'Nhóm fine-tune PhoBERT bằng cách nào?'. Trả lời: Nhóm dùng PhoBERT-base-v2 (huggingface: vinai/phobert-base-v2), gắn thêm tầng Linear Classifier trên vector đầu ra của token [CLS], huấn luyện với Learning Rate nhỏ (2e-5) và optimizer AdamW."
    },
    {
      id: "q10",
      topic: "Tiền xử lý Tiếng Việt",
      question: "Câu hỏi 10: Tiền xử lý tiếng Việt khác tiếng Anh ở điểm nào quan trọng nhất? Nhóm xử lý bài toán Word Segmentation ra sao?",
      intuitive: "Tiếng Anh phân cách từ bằng dấu cách ('machine learning' là 2 từ). Tiếng Việt là ngôn ngữ đơn lập, dấu cách chỉ phân cách các 'tiếng' (âm tiết), còn 'từ' có thể là từ đơn ('học') hoặc từ ghép ('học sinh', 'giáo viên'). Nếu không ghép từ lại bằng dấu gạch dưới ('giáo_viên'), máy tính sẽ hiểu 'giáo' là vũ khí còn 'viên' là viên thuốc!",
      academic: "Đặc trưng tiếng Việt là ranh giới từ vựng không trùng với khoảng trắng. Quá trình tiền xử lý bắt buộc phải có bước Phân đoạn từ (Word Segmentation). Nhóm sử dụng thư viện `pyvi` (hoặc vncorenlp) để ghép các âm tiết tạo thành từ ghép có nghĩa: 'giáo viên nhiệt tình' -> ['giáo_viên', 'nhiệt_tình']. Đồng thời chuẩn hóa mã Unicode dựng sẵn (NFC), chuyển chữ thường, loại bỏ ký tự rác và chuẩn hóa dấu thanh tiếng Việt.",
      gotcha: "Nếu quên tách từ ghép, từ vựng sẽ bị phân mảnh nghiêm trọng, làm giảm đáng kể hiệu năng của Word2Vec và LSTM."
    }
  ],

  // 8. Bộ câu hỏi trắc nghiệm tự đánh giá năng lực (Interactive Self-Assessment Quiz)
  quizQuestions: [
    {
      id: 1,
      question: "Giả thuyết phân bố (Distributional Hypothesis) của J.R. Firth (1957) phát biểu điều gì?",
      options: [
        "Từ nào dài hơn thì mang nhiều thông tin hơn",
        "Ý nghĩa của một từ được định hình bởi các từ thường xuất hiện cùng ngữ cảnh với nó",
        "Mỗi từ phải được mã hóa bằng một vector nhị phân độc lập",
        "Tần số xuất hiện của từ tuân theo phân phối Poisson"
      ],
      correct: 1,
      explanation: "Câu nói kinh điển của Firth: 'You shall know a word by the company it keeps' - Đây là nền tảng cốt lõi cho mọi mô hình Word Embedding hiện đại như Word2Vec, GloVe."
    },
    {
      id: 2,
      question: "Trong One-Hot Encoding với từ điển |V| = 50.000 từ, tích vô hướng giữa 2 vector của từ 'thầy_cô' và 'giảng_viên' bằng bao nhiêu?",
      options: [
        "0.85",
        "1.00",
        "0.00",
        "Không xác định được"
      ],
      correct: 2,
      explanation: "Trong One-Hot, mỗi từ chỉ có duy nhất một vị trí mang số 1, các vị trí còn lại bằng 0. Hai từ khác nhau sẽ có vị trí số 1 lệch nhau, do đó tích vô hướng e_i^T * e_j luôn luôn bằng 0 (hai vector hoàn toàn trực giao)."
    },
    {
      id: 3,
      question: "Kiến trúc nào của Word2Vec nhận TỪ TRUNG TÂM làm đầu vào để dự đoán CÁC TỪ NGỮ CẢNH xung quanh?",
      options: [
        "CBOW (Continuous Bag-of-Words)",
        "Skip-Gram",
        "TF-IDF",
        "FastText"
      ],
      correct: 1,
      explanation: "Skip-Gram lấy từ trung tâm w_t để dự đoán các từ ngữ cảnh w_{t+j}. Ngược lại, CBOW lấy các từ ngữ cảnh để dự đoán từ trung tâm."
    },
    {
      id: 4,
      question: "Tại sao trong Negative Sampling, phân phối Unigram lại được nâng lên lũy thừa 3/4 (0.75)?",
      options: [
        "Để tăng cơ hội được chọn cho các từ hiếm và giảm ưu thế của các từ quá phổ biến",
        "Để chuẩn hóa tổng xác suất bằng 1",
        "Để chuyển đổi từ số thực sang số nguyên",
        "Để tăng tốc độ tính toán ma trận trên GPU"
      ],
      correct: 0,
      explanation: "Lũy thừa 0.75 là nghiệm thực nghiệm tối ưu của Mikolov: nó kéo xác suất của các từ hiếm lên đáng kể so với tần số thật, giúp vector của chúng được cập nhật tốt hơn."
    },
    {
      id: 5,
      question: "Cổng nào trong LSTM chịu trách nhiệm quyết định lượng thông tin cũ trong Cell State cần bị loại bỏ?",
      options: [
        "Input Gate (Cổng vào)",
        "Output Gate (Cổng ra)",
        "Forget Gate (Cổng quên)",
        "Cell Gate (Cổng ô)"
      ],
      correct: 2,
      explanation: "Forget Gate f_t = sigma(W_f [h_{t-1}, x_t] + b_f) trả về giá trị trong [0, 1] để nhân trực tiếp với c_{t-1}, quyết định giữ hay xóa thông tin cũ."
    },
    {
      id: 6,
      question: "Trong đồ án này, vector biểu diễn văn bản v_D được trích xuất từ mô hình BiLSTM thuần túy bằng cách nào?",
      options: [
        "Lấy trung bình cộng tất cả các từ trong câu",
        "Ghép nối trạng thái ẩn bước cuối cùng của chiều xuôi và bước đầu tiên của chiều ngược: [h_T_right ; h_1_left]",
        "Chỉ lấy trạng thái ẩn của từ đầu tiên",
        "Dùng ma trận tích vô hướng ngẫu nhiên"
      ],
      correct: 1,
      explanation: "Công thức v_D = [h_T_right ; h_1_left] in R^{256} kết hợp trạng thái kết thúc của cả 2 chiều đọc, mang đầy đủ ngữ cảnh xuôi và ngược của cả câu."
    },
    {
      id: 7,
      question: "Trên tập dữ liệu Thương mại điện tử (E-Commerce), việc bổ sung cơ chế Self-Attention vào BiLSTM giúp tăng độ chính xác bao nhiêu %?",
      options: [
        "+0.83%",
        "+5.33% (từ 67.17% lên 72.50%)",
        "+15.84%",
        "Không thay đổi"
      ],
      correct: 1,
      explanation: "Trên E-Commerce, BiLSTM-Attention đạt bước nhảy vọt quan trọng từ 67.17% lên 72.50% (+5.33%, F1-score 72.40%) nhờ khả năng lọc và gán trọng số tập trung vào từ cảm xúc then chốt."
    },
    {
      id: 8,
      question: "Mô hình nào đạt độ chính xác cao nhất (SOTA) trên cả 2 bộ dữ liệu UIT-VSFC (95.50%) và E-Commerce (86.17%)?",
      options: [
        "TF-IDF + Logistic Regression",
        "Average Word2Vec",
        "BiLSTM Classifier",
        "PhoBERT-base-v2"
      ],
      correct: 3,
      explanation: "PhoBERT-base-v2 (Transformer 12 tầng tiền huấn luyện trên 20GB văn bản tiếng Việt) vượt trội áp đảo trên cả hai miền dữ liệu."
    },
    {
      id: 9,
      question: "Điểm yếu lớn nhất của Average Word2Vec là gì?",
      options: [
        "Tính toán quá chậm",
        "Mất thứ tự từ, đảo ngữ cho ra vector giống hệt nhau và không xử lý được phủ định kép",
        "Vector có số chiều quá lớn",
        "Không áp dụng được cho tiếng Việt"
      ],
      correct: 1,
      explanation: "Do phép trung bình cộng có tính giao hoán, các câu có cùng từ vựng nhưng đảo trật tự mang nghĩa trái ngược ('phim hay chứ không dở' vs 'phim dở chứ không hay') sẽ sinh ra vector y hệt nhau."
    },
    {
      id: 10,
      question: "Đặc trưng ngôn ngữ nào của tập E-Commerce khiến các mô hình học máy truyền thống gặp khó khăn?",
      options: [
        "Ngữ pháp quá chuẩn mực và câu quá dài",
        "Nhiều từ lóng, teencode, viết tắt, cấu trúc câu tự do và sắc thái châm biếm",
        "Hoàn toàn không có nhãn dữ liệu",
        "Dữ liệu chỉ toàn số không có chữ"
      ],
      correct: 1,
      explanation: "Tập E-Commerce chứa rất nhiều từ lóng ('ko', 'đc', 'ship'), teencode, icon cảm xúc và nhiều sắc thái châm biếm, dẫn đến hiện tượng dịch chuyển miền và tỷ lệ từ ngoài từ điển OOV cao."
    }
  ]
};
