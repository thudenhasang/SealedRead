# Nhật ký quyết định

Trạng thái: **chốt** (đã quyết), **đề xuất** (chờ nhóm duyệt), **chờ S0** (quyết bằng số đo).

| # | Quyết định | Trạng thái |
|---|---|---|
| D0 | Hash commit của `configs/prereg.yaml` trước S0: `________` | chờ S0 |
| D1 | **Hai tầng cài đặt.** Tầng 1 là vLLM gốc: SealedRead chạy hai luồng, luồng che ở khung rút gọn (Mục 3.7 và Phụ lục A); Không mask chạy một luồng, KV trùng khít chuỗi đơn. Tầng 2 là chuỗi đơn khung đầy đủ, hiện chạy bằng engine tham chiếu; plugin vLLM là spike E0. Hệ quả: với tầng 1, Δ_mask gộp tác dụng của che và dồn vị trí, nên phải nói rõ trong bài. | đề xuất |
| D2 | Bản sao `<locate>` có cùng *dải* vị trí và cùng độ dài với bản gốc, nội dung là các ID đã sắp theo giá trị. Đây là chỗ sửa câu "cùng token, cùng position ID" ở Mục 3.3. | chốt |
| D3 | Ở vòng R_max, `beta=reject_and_requery` dẫn sang `<reason>` (ép `alpha=answer`), không mở `<query>` mới. | chốt |
| D4 | Tokenization chuẩn theo mảnh ở mọi nơi. Engine tham chiếu mã hóa lại mỗi bước; vLLM dựng prompt từ văn bản. | chốt |
| D5 | Regex của vLLM không cấm được ID lặp, nên engine bỏ ID lặp và đếm (`stats.dropped_ids`). Engine tham chiếu dùng trie nên chính xác tuyệt đối. | chốt |
| D6 | RQ3: khi đã lọc Recall ≥ 0,5, biến thể mềm và cứng chỉ khác nhau ở cách phá hòa trong ≤ 4 mẫu, nên gần như trùng. Code giữ ba biến thể như đề xuất (`rest_em.VARIANTS`). Đề nghị thêm biến thể "mềm, không lọc recall, xếp hạng bằng R_μ" nếu muốn RQ3 có thông tin. | đề xuất |
| D7 | Kiểm định không-kém-hơn ở 32K với n = 1.000 và biên −1 điểm: SE ≈ 1,4 điểm, nên cận dưới CI chỉ qua −1 khi Δ ước lượng ≳ +1,8. Muốn power 80% khi Δ = 0 cần khoảng 15.700 câu (`stats.noninferiority_n`). `prereg.yaml` đang giữ giá trị của đề xuất; đề nghị nới biên lên −3 hoặc kiểm định gộp. | đề xuất |
| D8 | Chứng minh Tính chất 1: chặn H(P_k) ≤ B(N,K) cần điều kiện hóa thêm trên ánh xạ nhãn ID L (độc lập với nội dung). Không điều kiện hóa thì P_k mang tới log₂ Σ C(9999, j) ≈ 37 bit mỗi vòng. Chỉ sửa phần viết, không ảnh hưởng code. | đề xuất |
| D9 | Tự cài Qwen3 (`model/qwen3.py`) thay vì dùng transformers: cần mask theo từng token, position ID tùy chỉnh và attend một tập con của KV cache. Đã test khớp với `transformers.Qwen3ForCausalLM` (atol 1e-4). | chốt |
| D10 | Backbone Qwen3-4B-Instruct-2507 (full attention). Qwen3.5-4B (hybrid, mô hình của keychain-dv) bị loại vì điều kiện (i). `Qwen3Config.from_dict` từ chối kiến trúc hybrid. | chốt |
| D11 | DA-ft: khối `<knowledge>` rỗng, bằng chứng đi qua KV gốc của đoạn đã mở (mở từ bản sao của vòng chọn); bản sao vẫn có. | chốt |
| D12 | Chế độ oracle (P0 và so sánh trên cùng đoạn dán): G được dán ở `<locate>` đầu tiên, rồi R_max = 1 ép trả lời. Chạy zero-shot ở chế độ lenient: thiếu alpha/beta thì điền mặc định và đếm vào `stats.format_fixes`. | chốt |
| D13 | Tỉ lệ nhảy bước tính "văn bản pha che" gồm văn bản mô hình viết và văn bản đã dán trước lần `<locate>` đó. | đề xuất |
| D14 | Mọi đoạn Wikipedia multi-hop có Mục = `Introduction` (bộ dữ liệu không có thông tin mục); kiểm tra 3 xác nhận không lộ. | chốt |
| D15 | Bản tối giản của Bảng 7 không có pilot full FT (6–9 GPU-giờ), trong khi Mục 5.5 gọi pilot là bắt buộc. Cần cộng thêm vào bản tối giản. | đề xuất |
| D16 | Tên checkpoint teacher (`Qwen/Qwen3.6-27B-FP8` trong `configs/teacher.yaml`) và tên dataset benchmark trên Hugging Face **chưa kiểm chứng**. Kiểm tra ở lần chạy đầu. | chờ S0 |
| D17 | Đơn giá SFT ở Mục 5.5 (20–25 GPU-giờ cho "một lượt") chưa rõ là một epoch hay cả 2–3 epoch. Thay bằng số đo của `scripts/cost_probe.py --part sft`. | chờ S0 |
| D18 | Máy lớn: mỗi GPU một tiến trình vLLM 4B theo phần câu hỏi (`par.sh`); SFT bằng torchrun, gradient LoRA all-reduce thủ công; teacher tensor parallel 2. | đề xuất |
| D19 | Kế thừa từ keychain-dv: `config.py` (thêm biến môi trường), quy ước Colab (marker, persist, TIME_CAP), `tools/colab.py`, `fetch_data.py`, cách đọc shard. | chốt |
