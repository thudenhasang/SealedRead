# Kết quả thử nghiệm trên Colab (mẫu, điền sau khi chạy)

Ngày: ____ · GPU: ____ (40 / 80 GB) · vLLM: ____ · torch: ____ · commit: ____

## A. Kiểm tra code
| | Kết quả |
|---|---|
| pytest | __ passed |
| smoke.py | SMOKE_DONE: có / không |

## B. Diễn tập 0.6B
| Bước | Thời gian | Kết quả / ghi chú |
|---|---|---|
| build + check | | |
| teacher (số trace thật / bù giả lập) | | __ / __ |
| SFT sealedread / nomask | | loss đầu → cuối: |
| ReST-EM (số mẫu được chọn) | | |
| M-step | | |
| eval + report | | |
| Lỗi đã gặp và đã sửa (commit) | | |

## C. Đo đạc 4B
**Tokenizer:** PASS / FAIL · số token tối đa của `<locate>`: __

**Dữ liệu thử:**
| Kiểm tra | Giá trị | Ngưỡng | Đạt? |
|---|---|---|---|
| 1 ánh xạ G | | ≥ 0,99 | |
| 3 AUC bề mặt (thường / hư cấu) | / | ≤ 0,60 | |
| 4 không rò (32K / 64K / 128K) | / / | ≥ 0,99 | |
| 2 closed-book (multi-hop / hư cấu) | / | ≤ 0,15 / ≤ 0,02 | |
| thời gian dựng dữ liệu | | | |

**Parity (kiểm tra 6):**
| Cặp | Trung vị \|Δlog p\| | p99 | Đạt? |
|---|---|---|---|
| vLLM / ours, chuỗi thường (mốc) | | | |
| vLLM / ours, luồng che | | | |
| Flex / dense, chuỗi có mask (p99 ở token biên) | | | |

**Chi phí:**
| | 64K | 128K |
|---|---|---|
| SealedRead hai luồng: giây / episode | | |
| Không mask một luồng: giây / episode | | |
| Số vòng trung bình | | |
| LoRA trong vLLM có tác dụng | | |

| SFT 4B LoRA (4 trace / bước) | sealedread | nomask |
|---|---|---|
| giây / bước (bỏ bước compile) | | |
| VRAM đỉnh (GB) | | |
| độ dài chuỗi trung bình (token) | | |

**S0 thử nghiệm ở 64K (câu hỏi từ shard train; không phải cổng chính thức):**
| Cấu hình | Số câu | EM | format_fixes |
|---|---|---|---|
| (a) full attention | 100 | | – |
| (d) oracle, không che | 300 | | |
| (f) oracle, che, khung rút gọn (thiết kế chính) | 300 | | |
| (b) oracle, che, khung đầy đủ | 150 | | |

| Hiệu | Giá trị (điểm) | CI95 |
|---|---|---|
| EM(f) − EM(d) | | |
| Π = EM(f) − EM(b) | | |
| EM(f) − EM(a) | | |
| Recall@3 của `<locate>` zero-shot (100 câu) | | – |

`pilot_reading`: ____ → quyết định: làm tiếp / xem lại trace trước

**Teacher pilot (nếu có A100 80 GB):** trần p99: reason __ / query __ / recheck __ / answer __ · tỉ lệ qua lọc: __

## D. Ước lượng cho máy 8×B200
| Hạng mục | Từ số đo (A100) | Quy đổi B200 | So với Bảng 7–8 |
|---|---|---|---|
| Sinh mẫu ReST-EM (mỗi nhánh) | | | 5–8 GPU-giờ |
| SFT một run | | | 4–6 GPU-giờ |
| Đánh giá RQ1 (một mô hình, đủ ba mốc) | | | 4–7 GPU-giờ |
| **Tổng dự kiến** | | | 177–281 GPU-giờ |

Kết luận: lên máy lớn được chưa? ____ · việc cần làm trước: ____
