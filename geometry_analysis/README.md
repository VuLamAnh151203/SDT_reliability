# Phân tích geometry của hai Emotion Wheel loss

Thư mục này chạy ablation có kiểm soát giữa Euclidean, hypersphere và
Poincaré ball. Cả ba nhánh dùng chung:

- pure feature `H_TT/H_AA/H_VV` của SDT;
- ba projection head `Linear(1024, 16)` có cùng cách khởi tạo;
- SDT backbone, classifier, fusion và SDT loss ban đầu;
- fixed Emotion Wheel, Wheel Prototype loss và Wheel CPCC;
- optimizer, split, seed và hyperparameter còn lại.

Tham số duy nhất thay đổi giữa E/S/P là phép chiếu sau linear và hàm khoảng
cách. Các run Wheel dùng `tical-mode=observe`, vì vậy CA-KD, HypCPCC, TiCAL
fusion adjustment và COLD không tham gia objective.

## Ma trận thí nghiệm

| ID | Geometry | lambda prototype | lambda CPCC |
|---|---|---:|---:|
| B0 | Không | 0 | 0 |
| E-P | Euclidean | 0.1 | 0 |
| E-C | Euclidean | 0 | 0.05 |
| E-PC | Euclidean | 0.1 | 0.05 |
| S-P | Spherical | 0.1 | 0 |
| S-C | Spherical | 0 | 0.05 |
| S-PC | Spherical | 0.1 | 0.05 |
| P-P | Poincaré | 0.1 | 0 |
| P-C | Poincaré | 0 | 0.05 |
| P-PC | Poincaré | 0.1 | 0.05 |

## Chạy

Từ folder `SDT_new`:

```bash
# Smoke test một cấu hình
bash geometry_analysis/run_ablation.sh P-PC \
  --gpu-id 0 --epochs 1 --max-batches 1

# Một seed, toàn bộ 10 cấu hình
bash geometry_analysis/run_ablation.sh all --gpu-id 0

# Nhiều seed ghép cặp
SEEDS="2024 2025 2026 2027 2028" \
  bash geometry_analysis/run_ablation.sh all --gpu-id 0
```

Mặc định runner dùng IEMOCAP, train trên toàn bộ training dialogues, chọn
checkpoint bằng test weighted F1 giống protocol SDT gốc, dimension 16 và Wheel
temperature 1. Có thể thay đổi bằng biến môi trường:

```bash
DATASET=MELD WHEEL_TEMPERATURE=0.5 SEEDS="2024 2025 2026" \
  bash geometry_analysis/run_ablation.sh all --gpu-id 0
```

Các argument đặt cuối lệnh được chuyển thẳng vào `train.py` và override giá
trị mặc định, ví dụ `--feature-path`, `--batch-size`, `--epochs` hoặc
`--selection-protocol test`.

## Tổng hợp

```bash
python geometry_analysis/summarize.py \
  geometry_analysis/results/iemocap
```

Script tạo:

- `geometry_runs.csv`: từng seed/run;
- `geometry_summary.csv`: mean và standard deviation;
- `geometry_paired_deltas.csv`: chênh lệch theo cùng seed;
- `geometry_summary.md`: bảng đọc nhanh.

So sánh chính là `P-PC - S-PC` và `P-PC - E-PC`. Chỉ diễn giải geometry khi
các run dùng cùng dimension, seed, split và cùng budget chọn temperature.

## Temperature sweep

Khoảng cách của ba geometry có scale khác nhau. Nếu thực hiện temperature
sweep theo protocol test hiện tại:

```bash
for temperature in 0.25 0.5 1.0 2.0; do
  WHEEL_TEMPERATURE="$temperature" SEEDS="2024 2025 2026" \
    OUTPUT_DIR="geometry_analysis/results/iemocap_temp_${temperature}" \
    bash geometry_analysis/run_ablation.sh all --gpu-id 0
done
```

Giữ mỗi temperature trong một `OUTPUT_DIR` riêng để script tổng hợp không trộn
các temperature thành cùng một nhóm. Nếu cần protocol nghiêm ngặt không dùng
test để tune temperature hoặc checkpoint, đặt `SELECTION_PROTOCOL=validation`.

## Phân tích latent geometry từ checkpoint

Phân tích một checkpoint, không train lại:

```bash
bash geometry_analysis/run_diagnostics.sh \
  geometry_analysis/results/iemocap/<run>/best_checkpoint.pt \
  --gpu-id 0
```

Phân tích tất cả checkpoint Wheel trong một thư mục:

```bash
bash geometry_analysis/run_diagnostics.sh \
  geometry_analysis/results/iemocap \
  --gpu-id 0
```

Mỗi run tạo folder `geometry_diagnostics_test` chứa:

- `geometry_report.json` và `geometry_report.md`;
- `sample_metrics.csv`: prototype margin, radius/norm, off-plane ratio;
- `per_class_metrics.csv`: compactness, silhouette và lớp cạnh tranh gần nhất;
- `class_pair_metrics.csv`: khoảng cách giữa từng cặp cảm xúc;
- `distance_tiers.csv`: same/adjacent/middle/far;
- `knn_metrics.csv`: geometry-aware k-NN;
- `confusion_matrix.csv`;
- `geometry_dashboard.png` và `confusion_matrix.png`.

`geometry_report.md` có thêm bảng Pearson/Spearman giữa hyperbolic radius
(Poincaré) hoặc embedding norm (Euclidean/spherical) với confidence cuối của
SDT. File tổng hợp `geometry_diagnostics_test.csv` chứa các cột
`t/a/v_radial_confidence_pearson` và `t/a/v_radial_confidence_spearman`.
Riêng checkpoint Poincaré còn tạo `poincare_radius_confidence.png`: hàng trên
là scatter radius–confidence kèm mean ± SEM theo confidence bin; hàng dưới so
sánh phân bố radius của dự đoán đúng và sai. Các bin số liệu tương ứng được lưu
trong `poincare_radius_confidence_bins.csv`.

Khi đầu vào là cả thư mục, script còn tạo
`geometry_diagnostics_test.csv` để so sánh E/S/P trên cùng một bảng. B0 được
bỏ qua vì SDT baseline không có projection head của geometry experiment.

## Ablation radial, residual và hyperbolic metric

Runner `run_causal_ablation.sh` giữ nguyên SDT, hai Wheel loss
(`lambda_proto=0.1`, `lambda_cpcc=0.05`), dữ liệu, seed và mọi
hyperparameter khác. Chỉ projection/constraint thay đổi:

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| `P-FREE` | Poincaré 16D, radius tự do | Mốc so sánh chung |
| `P-FIXED` | Poincaré 16D, mọi điểm có radius cố định | Radius tự do có ích không? |
| `P-2D` | Poincaré 2D | 16 chiều có tốt hơn 2 chiều không? |
| `P-ZERORES` | Head Poincaré 16D nhưng đặt chiều 3–16 bằng 0 | Residual dimensions có ích không? |
| `E-FREE` | Euclidean 16D, norm tự do | Metric/map Poincaré có ích không? |

Chạy từng cấu hình với một seed:

```bash
cd SDT_new
bash geometry_analysis/run_causal_ablation.sh P-FREE --gpu-id 0
bash geometry_analysis/run_causal_ablation.sh P-FIXED --gpu-id 0
bash geometry_analysis/run_causal_ablation.sh P-2D --gpu-id 0
bash geometry_analysis/run_causal_ablation.sh P-ZERORES --gpu-id 0
bash geometry_analysis/run_causal_ablation.sh E-FREE --gpu-id 0
```

Hoặc chạy liên tiếp cả năm cấu hình:

```bash
SEEDS=2024 bash geometry_analysis/run_causal_ablation.sh all --gpu-id 0
python geometry_analysis/summarize_causal.py \
  geometry_analysis/results/causal_iemocap
```

`P-FIXED` mặc định dùng radius 0.75. Có thể đổi bằng
`FIXED_RADIUS=0.5`. Kết quả tổng hợp gồm `causal_runs.csv`,
`causal_deltas.csv` và `causal_summary.md`. Bốn chênh lệch được tính theo
cùng seed:

```text
radial             = P-FREE - P-FIXED
dimension          = P-FREE - P-2D
residual           = P-FREE - P-ZERORES
hyperbolic_metric  = P-FREE - E-FREE
```

`P-2D` đồng thời giảm số tham số của projection head. Vì vậy,
`P-FREE - P-ZERORES` là kiểm soát trực tiếp hơn cho vai trò của các chiều
residual: cả hai vẫn dùng head 16D, nhưng `P-ZERORES` không cho output sử dụng
các chiều 3–16. So sánh `P-FREE - E-FREE` đo toàn bộ ảnh hưởng của
Poincaré map và metric trong implementation này; riêng một so sánh đó chưa đủ
để tách curvature khỏi khác biệt của phép map.
