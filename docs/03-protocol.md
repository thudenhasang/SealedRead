# 03 · Giao thức, token và mask

## Thẻ và máy trạng thái (`protocol.py`)
```
<reason>…alpha=query|answer</reason>  →  query → <locate> → <knowledge> (engine) → <recheck>…beta=…</recheck>
                                      →  answer (kết thúc)
beta: valid_to_think | explore_query → <reason>;  reject_and_requery → <query> (trừ ở vòng R_max: → <reason>, D3)
```
- Ở vòng R_max, `<reason>` chỉ được kết thúc bằng `alpha=answer`. Engine ép bằng cách sửa alpha cuối (`force_alpha_answer`).
- `validate()` liệt kê mọi vi phạm của Mục 5.1, gồm cả trần token theo tokenizer của học sinh.
- Đoạn đã dán bị loại khỏi ngữ pháp của các vòng sau.
- Engine tạo `<knowledge>` từ các ID mới, theo thứ tự trong C.

## Tokenization chuẩn (D4)
Mọi chuỗi token được ghép từ **mảnh**: thẻ, nội dung, `"\n"`, từng nhãn `[[S_k]]`, dấu cách giữa các nhãn. Mỗi mảnh tokenize riêng rồi nối lại.
- Hệ quả 1: trainer, engine tham chiếu và vLLM có cùng một chuỗi token cho mỗi trace.
- Hệ quả 2: bản sao `<locate>` (ID đã sắp) luôn dài bằng bản gốc, nên dùng chung được position ID.
- Engine tham chiếu: sau khi đóng thẻ, cuộn lại các token vừa sinh và mã hóa lại theo mảnh.
- vLLM: prompt luôn dựng lại từ văn bản.
- `scripts/tokenizer_check.py` kiểm tra các giả định này trên tokenizer thật.

## Bố cục token (`layout.py`)
Thứ tự lưu trữ (bằng thứ tự KV) của một chuỗi:
```
[A][Q][C][Q'][T'] GEN… [LOC₁][CPY₁][KNOW₁] GEN… [LOC₂][CPY₂][KNOW₂] … GEN <answer>…</answer><|im_end|>
```
| Loại | Là gì | Attend |
|---|---|---|
| A, Q | hướng dẫn, câu hỏi | nhân quả |
| C | ngữ cảnh | A, Q, C (prefill nhân quả) |
| QP, TP | Q' và template | A, Q, lịch sử (không C) |
| GEN | thẻ pha che | A, Q, Q', T', GEN, CPY, KNOW đứng trước (không C, không LOC) |
| LOC (vòng k) | `<locate>` gốc | A, Q, **C**, lịch sử, LOC của chính vòng k |
| CPY | bản sao `<locate>` | như GEN (ở Không mask thì thêm C) |
| KNOW | đoạn dán | như GEN |

**Theo nhánh:**
- `nomask`: thêm C cho mọi token sau C. Quy tắc ẩn LOC giữ nguyên.
- `daft`: token sau C thấy các token C của đoạn đã được chọn, kể từ bản sao của vòng chọn chúng. Khối dán rỗng.
- Ablation: `hide_locate=False` (kênh 4 mở); `locate_history="query_only"` (`<locate>` chỉ đọc `<query>` của nó).

**Position ID:**
- C giữ vị trí; mọi token sau C nối tiếp.
- Bản sao dùng lại vị trí của bản gốc; đoạn dán tiếp ngay sau vị trí sinh hiện tại.
- View `masked`, khung `full`: bỏ token C nhưng vẫn nhảy |C| vị trí (cấu hình (b) của P0, không cần prefill C).
- Khung `compact` (Phụ lục A): không nhảy.

**Target của loss:** GEN và LOC (view `chain`), chỉ GEN (view `masked`), chỉ LOC (view `locate`). Không tính trên CPY, KNOW, prompt.

## Mã thẻ trong code
`A=0 Q=1 C=2 QP=3 TP=4 GEN=5 LOC=6 CPY=7 KNOW=8 PAD=9`. Quy tắc dense (`layout.visible`, numpy) và quy tắc Flex (`model/attention.make_mask_mod`, torch) được test là cho ra cùng ma trận ở mọi nhánh và ablation.
