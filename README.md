# APL-MDD: reimplementation và ablation

Reimplement mô hình **APL** (Ye et al., *An Approach to Mispronunciation Detection and Diagnosis with Acoustic, Phonetic and Linguistic (APL) Embeddings*, ICASSP 2022) trên TIMIT + L2-ARCTIC.

Repo dùng để chạy ablation **AL / P / PL / APL**. Phonetic embedding lấy từ XLS-R-300M ở hai trạng thái:
* **raw**: model pretrained gốc, chưa finetune.
* **ft**: model đã finetune cho phoneme recognition trên đúng tập train của APL, bằng [XLSR-300M-Phoneme-Recognition](https://github.com/VietCH57/XLSR-300M-Phoneme-Recognition).

## Kiến trúc

Chi tiết từng phần, kèm số mục và số phương trình trong bài, nằm trong docstring của `apl/model.py`.

```
fbank 81 (10 ms) ─► E_a: 2×[Conv2D→BN→ReLU→Dropout] → 4×[BiLSTM→BN→Dropout] ─► H^a (T', 768) ┐
XLS-R emb (20 ms → lặp ×2 = 10 ms) ─► E_p: cùng 2 CNN stack → 1×[BiLSTM→BN→Dropout] ─► H^p (T', 768) ┤ concat → H^Q (T', 1536)
canonical phones ─► E_l: Embedding(512) → BiLSTM(384×2) = H^V → Linear(no bias) = H^K (1536)        │
                                                                                                   ▼
            α = softmax_n(h^Q · h^K)   c = Σ α h^V   y = softmax(W [c; h^Q] + b)   → CTC loss / beam search
```

| Biến thể | Query H^Q | Linguistic encoder |
|---|---|---|
| APL | [H^a; H^p] | có |
| PL | H^p | có |
| AL | H^a | có (= Baseline-2 / [14]) |
| P | H^p | không, đầu ra y = softmax(W h^Q + b) |

Đầu ra gồm 69 lớp: 39 âm vị chuẩn, 28 âm vị có dấu `*`, `err`, và blank `" "` (`configs/vocab.json`).

## Nguồn của từng siêu tham số

Mỗi giá trị trong `configs/apl.yaml` đều ghi rõ nguồn:
* **[APL]**: số liệu bài báo nêu trực tiếp.
* **[14]**: lấy từ code công khai của baseline mà APL dùng lại (bài viết: "parameters ... are the same as baselines"). Code ở [cageyoko/CTC-Attention-Mispronunciation](https://github.com/cageyoko/CTC-Attention-Mispronunciation).

Những điểm bài APL không nói và cách repo xử lý:

* **Kích thước CNN/LSTM, embedding 512, dropout 0.2, optimizer, lịch giảm learning rate, beam 10**: theo [14].
* **Chuẩn hoá fbank**: global CMVN tính trên tập train, theo [14]. Embedding XLS-R được giữ nguyên giá trị.
* **Canonical và Transcript** được dùng đúng như trong CSV, cho cả TIMIT và L2-ARCTIC.
* **Ghép số frame**: embedding XLS-R (20 ms) được lặp mỗi frame 2 lần rồi cắt hoặc pad cho bằng số frame T của fbank, đúng yêu cầu "identical number of frames" ở mục 2.2.

Các điểm thuần về cài đặt, không đổi mô hình:
* Packed LSTM.
* BatchNorm sau BiLSTM bỏ qua frame padding khi tính thống kê.
* Mask key padding trong attention.
* Batch được gom theo độ dài.
* Tính toán bf16 trên GPU (`--amp none` để chạy fp32).

Metric MD&D (`apl/metrics.py`) dùng đúng cách đếm của script chấm điểm trong [14], tức cấu trúc phân cấp của [16]. Phần căn chỉnh dùng thuật toán của Kaldi `align-text`. Trên các file kết quả đi kèm repo [14], tổng lỗi PER khớp với Kaldi: 4423/29087 và 7934/29087.

## GPU dùng chung

Mọi script dùng GPU đều có hai flag, mặc định giới hạn ở 50%, cơ chế giống repo XLSR-300M-Phoneme-Recognition:

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--gpu_memory_fraction` | 0.5 | Giới hạn VRAM của process (`torch.cuda.set_per_process_memory_fraction`). Vượt quá sẽ báo OOM thay vì lấn sang phần của người khác. 0 = không giới hạn |
| `--gpu_duty_pct` | 50 | Sau mỗi batch, process ngủ để GPU chỉ bận tối đa 50% thời gian thực. 100 = không giới hạn |

Các script áp dụng: `train.py` (lấy từ `configs/apl.yaml`), `evaluate.py`, `extract_xlsr.py` (flag dòng lệnh) và bước finetune XLS-R (`configs/xlsr_finetune.yaml`). Với duty 50%, thời gian chạy tăng khoảng gấp đôi.

## Cài đặt

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q
```

## Quy trình

`AUDIO_ROOT` là thư mục chứa audio, sao cho `$AUDIO_ROOT/<Path>.wav` tồn tại. Ví dụ `L2_arctic_WAV/ABA_arctic_a0003.wav` hoặc `train/dr6/mabc0/sa1.wav`. Đuôi `.WAV`, chữ hoa kiểu TIMIT gốc và định dạng NIST SPHERE đều đọc được.

```bash
AUDIO_ROOT=/path/to/wavs

# 0. Kiểm tra dữ liệu: số câu, speaker, vocab, file thiếu
python scripts/check_data.py --audio_root $AUDIO_ROOT

# 1. Trích fbank 81 chiều và CMVN
python scripts/extract_fbank.py --audio_root $AUDIO_ROOT --out_dir feats/fbank

# 2. Finetune XLS-R-300M cho phoneme recognition, chỉ trên tập train của APL
python scripts/make_xlsr_finetune_data.py --audio_root $AUDIO_ROOT --out_dir xlsr_ft_data
#    Chạy trong repo XLSR-300M-Phoneme-Recognition:
python scripts/train.py --config /path/to/apl-mdd/configs/xlsr_finetune.yaml \
    --train_csv /path/to/apl-mdd/xlsr_ft_data/train.csv --dev_csv /path/to/apl-mdd/xlsr_ft_data/dev.csv \
    --vocab_path /path/to/apl-mdd/xlsr_ft_data/vocab.json --output_dir runs/apl_ft

# 3. Trích phonetic embedding (output layer cuối, bỏ CTC head), XLS-R đóng băng
python scripts/extract_xlsr.py --model facebook/wav2vec2-xls-r-300m --audio_root $AUDIO_ROOT --out_dir feats/xlsr_raw
python scripts/extract_xlsr.py --model /path/to/XLSR-300M-Phoneme-Recognition/runs/apl_ft/best \
    --audio_root $AUDIO_ROOT --out_dir feats/xlsr_ft

# 4. Ablation: 7 cấu hình, mỗi cấu hình 1 run với seed 42, kết quả tổng hợp ở runs/results_test.md
bash scripts/run_ablation.sh
```

Chạy riêng một cấu hình:

```bash
python scripts/train.py --variant APL --phon_dir feats/xlsr_ft --out_dir runs/APL-ft
python scripts/evaluate.py --run_dir runs/APL-ft --split test --beam 10
```

Mỗi run ghi ra các file sau:
* `train.log`, `history.json`: log và lịch sử huấn luyện.
* `best.pt`: model có dev PER tốt nhất (greedy decoding), theo cách chọn của [14].
* `{dev,test}_metrics.json`: đủ các cột của Bảng 2, cộng TA/FR/FA/CD/DE và kết quả tách theo loại lỗi sub/ins/del.
* `{dev,test}_decoded.jsonl`: canonical, human và pred của từng câu.

## Cấu trúc

```
apl/            vocab, manifest (CSV), audio, features (fbank/CMVN), dataset,
                model, decoding (greedy/beam), metrics, engine
scripts/        check_data, extract_fbank, make_xlsr_finetune_data, extract_xlsr,
                train, evaluate, summarize, run_ablation.sh
configs/        apl.yaml, vocab.json, xlsr_finetune.yaml
data/           train.csv (TIMIT 6300 + L2-ARCTIC 1800), dev.csv (899), test.csv (900)
tests/          pytest
```
