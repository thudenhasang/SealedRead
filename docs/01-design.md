# 01 · Thiết kế

## Ý tưởng
Thiết kế tách việc **đọc** khỏi việc **suy luận**:
- Phần suy luận diễn ra khi C bị che hoàn toàn.
- Chỉ token của `<locate>` được nhìn C, và chỉ được xuất 1–3 ID (ngữ pháp ép).
- Engine dán nguyên văn các đoạn được chọn, theo thứ tự xuất hiện trong C.
- Câu hỏi được lặp lại ngay sau C (bố cục kẹp) để đưa nó về gần phần suy luận.

Nhờ vậy, thông tin từ C chỉ đến được phần suy luận qua văn bản đã khai báo, cộng với một lượng bit có chặn trên qua việc chọn ID (Tính chất 1).

## Một nguồn sự thật cho mask
`srd/layout.py` dựng chuỗi token của một trace và định nghĩa quy tắc attend `visible()`. Mọi thành phần đều lấy từ đây:

| Thành phần | Dùng layout thế nào |
|---|---|
| Trainer (`train/sft.py`) | Mask Flex sinh từ cùng quy tắc (`model/attention.make_mask_mod`). Test kiểm tra hai bản cho ra cùng ma trận |
| Engine tham chiếu (`engine/ref.py`) | Dựng chuỗi bằng cùng `SeqBuilder`, mask từng bước bằng `visible()`. Test kiểm tra trace nó sinh ra cho đúng token và mask như `build_seq` |
| Engine vLLM (`engine/vllm_policy.py`) | Prompt lấy từ `Seq.prompt_ids()`: bỏ bản gốc `<locate>` đã ẩn, giữ bản sao đúng vị trí |
| Phép thử (`eval/probes.py`) | Chạy forward toàn chuỗi với đúng mask của nhánh |

## Cài đặt: hai luồng (D1)
- **Luồng suy luận** không chứa C: `[A][Q][Q'][T']` cộng lịch sử, vị trí liền sau Q (khung rút gọn, Phụ lục A cũ). Viết `<reason>`, `<query>`, `<recheck>`, `<answer>`.
- **Luồng `<locate>`** chứa C: `[A][Q][C][Q'][T']` cộng lịch sử. Chỉ viết được ID (ngữ pháp). vLLM cache phần `[A][Q][C]` nên mỗi câu chỉ tính C một lần.
- Engine dán đoạn được chọn vào lịch sử chung của hai luồng.
- Hai luồng **tương đương về thông tin** với một chuỗi: luồng suy luận không hề chứa C, nên kênh rò 1–4 bị bịt theo cấu trúc.
- Huấn luyện khớp từng token: nhánh `sealedread` dùng view `masked` (compact) cho các thẻ che, cộng view `locate` cho `<locate>`.
- Nhánh Không mask chạy một luồng có C, khung đầy đủ. Bỏ bản gốc `<locate>`, giữ bản sao tại chỗ thì KV *trùng khít* với chuỗi đơn.
- **Hệ quả:** Δ_mask đo tác dụng gộp của việc che và khoảng cách vị trí ngắn. P0 cấu hình (b) (cùng mask, khung đầy đủ) đo phần do vị trí (Π) để báo cáo kèm.
- Engine tham chiếu (một chuỗi, mọi nhánh, chậm) chỉ dùng cho test, cho P0 (b), và cho DA-ft tùy chọn.

## Các nhánh (Bảng 10)
| Nhánh | Pha che thấy C? | Bằng chứng | Huấn luyện (view) | Suy luận |
|---|---|---|---|---|
| `sealedread` (chính) | Không | đoạn dán, mã hóa khi che C | masked(compact) + locate | vLLM hai luồng |
| `sealedread_chain` (không chạy) | Không | như trên, khung đầy đủ | chain | engine tham chiếu |
| `nomask` | Có | đoạn dán, mã hóa khi thấy C | chain | vLLM một luồng |
| `daft` (tùy chọn, D22) | Không, trừ đoạn đã mở | KV gốc của các đoạn đã chọn (khối dán rỗng) | chain | engine tham chiếu |
| `rag` | Không | retriever thay `<locate>` | masked(compact) | vLLM, luồng che |
| `sealong` | Có (full attention) | không có giao thức | plain | vLLM |

## Tính chất 1 trong code
Các điều kiện (i)–(vii) của Tính chất 1 tương ứng với code như sau:
- (i) Backbone chỉ có attention: `Qwen3Config.from_dict` từ chối kiến trúc hybrid hoặc sliding window.
- (ii), (iii) Quy tắc của `visible()`.
- (iv) Ngữ pháp (`grammar.py`) cộng bản sao đã sắp (`layout.copy_pieces`).
- (v) Engine dán (`layout.add_step`).
- (vi) ID ngẫu nhiên (`data/ids.py`).
- (vii) Position ID trong `SeqBuilder`.

Phép thử bất biến (`test_model.test_invariance_*`, `scripts/probes.py`) là **kiểm thử cài đặt**. Với SealedRead, thay mọi token C ngoài các đoạn đã dán thì log-prob ở pha che không đổi (bằng 0 tuyệt đối trên CPU). Với Không mask thì đổi. Với DA-ft cũng đổi, vì KV gốc của đoạn đã mở mang ngữ cảnh của các đoạn đứng trước (kênh 2, đo bằng Δ_ctx).

Chứng minh cần điều kiện hóa thêm trên ánh xạ nhãn ID (D8).
