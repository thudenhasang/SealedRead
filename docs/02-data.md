# 02 · Dữ liệu

## Nguồn
Dùng cùng shard với keychain-dv, ở `MyDrive/ragproj2/data/train/`:
- `<dataset>_s<k>.json` là các bản ghi chính thức của HotpotQA (distractor), 2WikiMultihopQA và MuSiQue.
- s0–s4 (và s10–s14 của MuSiQue) dùng cho huấn luyện; s90 dùng cho dev/test.
- `scripts/fetch_data.py` giữ nguyên từ repo cũ: `--from-drive`, hoặc `--from-hf` (cùng revision đã pin).

`data/qa.py` chuyển mỗi bản ghi thành `QAItem`:
- **Câu hỗ trợ ở mức câu**: dùng `supporting_facts` để gán đoạn chuẩn chính xác sau khi chia.
- **Thứ tự bước (`hops`)** cho tỉ lệ nhảy bước:
  - MuSiQue: `question_decomposition`;
  - 2Wiki: bộ ba bằng chứng (loại compositional/inference);
  - HotpotQA bridge: đoạn nhắc tên đoạn kia đứng trước.
- **Loại đáp án**: *đóng* (có/không, một trong hai thực thể được so sánh, số đếm ≤ 10) hoặc *mở*.

## Một mẫu được dựng thế nào (`data/build.py`)
1. Đưa vào các đoạn của chính câu hỏi: đoạn chuẩn luôn có; đoạn nhiễu của nó chỉ có nếu qua bộ lọc rò.
2. Thêm nhiễu khó bằng BM25 trên kho đoạn, truy vấn = câu hỏi + tiêu đề đoạn chuẩn; không trùng tiêu đề, phải qua bộ lọc rò.
3. Thêm filler ngẫu nhiên cho đến khi đủ ngân sách token của C. Ngân sách tính cả nhãn và tiêu đề; đếm bằng tokenizer Qwen3 khi dựng thật.
4. Xáo thứ tự tài liệu (đoạn chuẩn ở vị trí ngẫu nhiên). Mỗi tài liệu chia đoạn ≤ 256 token theo ranh giới câu, gộp đoạn < 32 token trong cùng tài liệu.
5. Gán ID ngẫu nhiên trong 1–9999, tránh mọi số đứng độc lập trong văn bản (D: `S_1961` cạnh "born 1961"). G là các đoạn chứa câu hỗ trợ.

**Lọc rò (`data/leak.py`)**, theo loại đáp án:
- (a) Đáp án không có trong câu hỏi: loại đoạn nhiễu chứa đáp án, hoặc nêu lại sự kiện cầu nối (chủ thể của một bước đi cùng thực thể mà bước đó tiết lộ).
- (b) Đáp án có/không, hoặc chọn một trong hai thực thể: loại đoạn nêu chính thuộc tính được so sánh của các thực thể đó. Với HotpotQA (không có bộ ba bằng chứng) thì loại mọi đoạn nhiễu nhắc tới thực thể được so sánh.
- Với câu đếm/liệt kê, giá trị chung (ví dụ tên đạo diễn) không được xuất hiện trong đoạn nhiễu.

## Năm loại câu (Bảng 4, `data/splits.py`)
| Loại | Tỉ lệ | Cách làm |
|---|---|---|
| multihop | 50% | như trên |
| paraphrase | 15% | ghi kèm `meta.needs_paraphrase`. `scripts/paraphrase.py` (GPU) viết lại câu hỏi, giữ tên riêng, rồi chỉ giữ câu nếu đọc đoạn chuẩn vẫn ra đúng đáp án; không đạt thì trả về multihop |
| fictional | 15% | `data/fiction.py`: đổi tên nhất quán trong cả mẫu (câu hỏi, đáp án, mọi đoạn), tên mới ghép từ âm tiết, giữ độ dài và cách viết hoa |
| multiseg | 10% | câu so sánh / bridge-comparison của 2Wiki, cộng câu đếm/liệt kê dựng từ 3–6 đoạn 2Wiki có chung (quan hệ, giá trị) (`data/synth.py`) |
| nocontext | 10% | số học, logic, đếm chữ tự chứa, G = ∅. Vẫn có C để mô hình học *không* mở ngữ cảnh |

## Các tập (`configs/data.yaml`)
| File | Nguồn | Số câu |
|---|---|---|
| `train.jsonl` | shard train | 7.000, độ dài ngẫu nhiên 16K–24K |
| `val.jsonl` | shard train (không trùng) | 1.500 |
| `reserve.jsonl` | shard train | 3.000 (bù hạn ngạch cho teacher) |
| `dev_{32,64,128}k.jsonl` | s90 | 500 / 1.000 / 1.000 (P0, điểm quyết định sau S1) |
| `test_{64,128}k.jsonl`, `test_32k.jsonl` | s90 (không trùng dev) | 2.000 (cùng câu ở hai mốc); 1.000 ở 32K |
| `test_fict_{64,128}k.jsonl` | bản hư cấu của test | 2.000 |

Lọc closed-book chỉ áp dụng cho dữ liệu huấn luyện: `scripts/closed_book.py` trên shard sinh file, rồi đặt `build.closed_book_filter`. Mỗi cụm bootstrap là một câu hỏi gốc (`Sample.cluster`).

## Kiểm tra (`scripts/check_data.py`, ngưỡng ở `configs/prereg.yaml`)
| # | Kiểm tra | Ở đâu |
|---|---|---|
| 1 | Câu bằng chứng ánh xạ vào G ≥ 99% | `checks.check_gold_mapping` |
| 2 | Closed-book ≤ 10–15% (multi-hop), ≤ 2% (hư cấu) | `scripts/closed_book.py` (GPU) → `--closed-book` |
| 3 | Bộ phân loại bề mặt AUC ≤ 0,60. Với tập hư cấu có thêm n-gram ký tự của tiêu đề | `checks.check_surface` (hồi quy logistic, k-fold theo mẫu) |
| 4 | Không rò ≥ 99% ở từng mốc độ dài | `checks.check_leaks` |
| 5 | Diễn đạt lại giữ đáp án ≥ 95%, độ trùng từ giảm | `checks.check_paraphrase`, cộng 200 câu kiểm tra tay |
| 6 | Khớp log-prob giữa các engine | `scripts/parity.py` (GPU) |
| + | ID duy nhất, trong khoảng, không trùng số trong văn bản; mọi mẫu giải được qua giao thức bằng agent giả lập | `check_structure`, `check_scripted` |

Mọi đoạn Wikipedia ở đây đều là đoạn mở đầu, nên trường "Mục" được gán đồng loạt là `Introduction` (D14). Kiểm tra 3 xác nhận trường này không làm lộ nhãn.
