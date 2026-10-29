# Poincare mechanism on pure SDT features

Experiment này chỉ dùng `H_TT/H_AA/H_VV` và trả lời năm câu hỏi:

1. `E-E` so với `B-E` so với `B-P`: lợi ích đến từ ball map hay metric?
2. B0/E-E/S-S/B-P: Wheel supervision thay đổi chính pure representation ra sao?
3. Gradient Wheel có cùng hướng với SDT loss tại `H_mm` không?
4. Poincare sửa loại lỗi nào theo khoảng cách Emotion Wheel?
5. Poincare metric khuếch đại khoảng cách theo radius như thế nào?

## Cấu hình

| ID | Map | Distance | Wheel losses |
|---|---|---|---|
| B0 | none | none | 0 |
| P-0 | ball | Poincare | 0 (projector control) |
| E-E | linear | Euclidean | 0.1 / 0.05 |
| B-E | Exp-map to ball | Euclidean | 0.1 / 0.05 |
| S-S | unit sphere | angular | 0.1 / 0.05 |
| B-P | Exp-map to ball | Poincare | 0.1 / 0.05 |

Tất cả Wheel run dùng cùng SDT backbone, dimension 16, fixed semantic
prototypes radius 0.75, temperature 1 và `wheel-feature-source=pure`.

## Chạy train

Từ `SDT_new`:

```bash
# Smoke test toàn pipeline
EPOCHS=1 bash experiment/poincare_pure_mechanism/run_train.sh all \
  --gpu-id 0 --max-batches 1 \
  --output-dir experiment/poincare_pure_mechanism/results/smoke

# Một seed, protocol test-selected giống kết quả 75.31 trước đây
bash experiment/poincare_pure_mechanism/run_train.sh all --gpu-id 0

# Nhiều seed (khuyến nghị để kết luận về chênh lệch nhỏ B-P và E-E)
SEEDS="2024 2025 2026 2027 2028" \
  bash experiment/poincare_pure_mechanism/run_train.sh all --gpu-id 0
```

Chạy riêng một cấu hình:

```bash
bash experiment/poincare_pure_mechanism/run_train.sh B-E --gpu-id 0
```

Để dùng validation checkpoint selection:

```bash
SELECTION_PROTOCOL=validation \
  bash experiment/poincare_pure_mechanism/run_train.sh all --gpu-id 0
```

## Phân tích

Sau khi tất cả run hoàn thành:

```bash
bash experiment/poincare_pure_mechanism/run_analysis.sh \
  experiment/poincare_pure_mechanism/results/train \
  --device cuda --gpu-id 0
```

Kết quả nằm tại `experiment/poincare_pure_mechanism/results/analysis`:

- `performance.csv`: E-E/B-E/B-P và các control;
- `causal_deltas.csv`: đóng góp ball map và hyperbolic metric theo từng seed;
- `pure_geometry.csv`: Fisher ratio, centroid margin, kNN@5, silhouette và
  semantic Spearman trực tiếp trên `H_mm`;
- `gradient_alignment_samples.csv` và `gradient_alignment_summary.csv`;
- `error_transitions.csv`: Wrong-to-Correct và Correct-to-Wrong theo Wheel tier;
- `distance_amplification.csv`: tỷ số `d_P/d_E` theo radius và Wheel tier;
- `REPORT.md` cùng ba hình tổng hợp.

Hai hiệu chính cần đọc đầu tiên:

```text
ball-map contribution = B-E - E-E
metric contribution   = B-P - B-E
```

`run_analysis.sh` không retrain và không thay đổi checkpoint.
Nếu có nhiều run trùng ID/seed, analysis dùng checkpoint mới nhất và in rõ run
bị bỏ qua, tránh trộn các lần chạy lặp ngoài ý muốn.

Error transition ưu tiên dùng `P-0` làm reference vì đây là control có cùng
projector/RNG footprint nhưng không có Wheel gradient; nếu thiếu `P-0`, script
mới fallback về B0.

