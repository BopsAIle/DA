# Phân tích Dataset: GreenNode/zalo-ai-legal-text-retrieval-vn

## 1. Tổng quan

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên dataset** | GreenNode/zalo-ai-legal-text-retrieval-vn |
| **Tổ chức** | GreenNode |
| **Loại task** | Text Retrieval, Document Retrieval |
| **Ngôn ngữ** | Tiếng Việt (Vietnamese) |
| **Lĩnh vực** | Pháp luật (Legal) |
| **License** | MIT |
| **Kích thước** | 10K - 100K (63,036 rows) |
| **Dung lượng** | 150 MB |

## 2. Cấu trúc Dataset

Dataset bao gồm 4 subsets chính:

| Subset | Số dòng | Mô tả |
|--------|---------|-------|
| `defaultcorpus` | 61,400 | Corpus văn bản pháp luật |
| `queries` | 818 | Các câu truy vấn |
| `qrels` | 793 | Relevance judgments (ánh xạ query -> document) |
| `default` | - | Split mặc định |

### 2.1 Splits

- **train**: Dữ liệu huấn luyện
- **test**: Dữ liệu kiểm tra

## 3. Thống kê chi tiết (trên tập test)

### 3.1 Thống kê Documents (văn bản pháp luật)

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số ký tự | 83,463,474 |
| Số lượng documents | 62,243 |
| Độ dài nhỏ nhất | 16 ký tự |
| Độ dài trung bình | 1,358.79 ký tự |
| Độ dài lớn nhất | 252,963 ký tự |
| Số documents duy nhất | 60,701 |

**Nhận xét**: Độ dài trung bình ~1.36K ký tự cho thấy đây là các văn bản pháp luật ngắn đến trung bình (khoảng 200-250 từ).

### 3.2 Thống kê Queries (câu truy vấn)

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số ký tự | 68,536 |
| Số lượng queries | 818 |
| Độ dài nhỏ nhất | 16 ký tự |
| Độ dài trung bình | 83.78 ký tự |
| Độ dài lớn nhất | 164 ký tự |
| Số queries duy nhất | 788 |

**Nhận xét**: Các truy vấn khá ngắn, trung bình ~84 ký tự (khoảng 12-15 từ), phù hợp với việc tìm kiếm văn bản pháp luật.

### 3.3 Thống kê Relevant Documents

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số relevant docs | 793 |
| Số relevant docs nhỏ nhất/query | 1 |
| Số relevant docs trung bình/query | 1.006 |
| Số relevant docs lớn nhất/query | 2 |
| Số unique relevant docs | 447 |

**Nhận xét**: Mỗi query thường chỉ có 1 document liên quan, cho thấy đây là task **retrieval** (tìm kiếm văn bản) chứ không phải reranking.

## 4. Bối cảnh và nguồn gốc

- **Nguồn gốc**: Dataset đến từ cuộc thi **Zalo AI Challenge** về bài toán tìm kiếm văn bản pháp luật tiếng Việt
- **Tham chiếu**: https://challenge.zalo.ai/
- **Tích hợp**: Được tích hợp vào **MTEB** (Massive Text Embedding Benchmark) với tên task là `ZacLegalTextRetrieval`

## 5. Các paper liên quan

| Paper | Năm | Mô tả |
|-------|-----|-------|
| MMTEB: Massive Multilingual Text Embedding Benchmark | 2025 | Benchmark đa ngôn ngữ |
| MTEB: Massive Text Embedding Benchmark | 2022 | Benchmark gốc |
| GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task | 2026 | Benchmark tiếng Việt |

## 6. Sử dụng với MTEB

```python
import mteb

task = mteb.get_task("ZacLegalTextRetrieval")
evaluator = mteb.MTEB([task])

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

## 7. Đặc điểm nổi bật

1. **Ngôn ngữ**: Tiếng Việt trong lĩnh vực pháp luật - đòi hỏi hiểu biết về thuật ngữ pháp lý
2. **Task type**: Dense Retrieval (tìm kiếm văn bản dựa trên embedding)
3. **Độ khó**: Trung bình - mỗi query chỉ có 1-2 documents liên quan
4. **Kích thước phù hợp**: đủ lớn để fine-tune, đủ nhỏ để thử nghiệm nhanh

## 8. Metrics đánh giá (MTEB)

Thông thường cho task retrieval:
- **nDCG@10**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Recall@100**: Tỷ lệ recall ở top 100
- **MRR**: Mean Reciprocal Rank
