# Emotion Wheel mechanism experiments

Thư mục này kiểm tra **vì sao** hai Emotion Wheel loss cải thiện SDT. Ablation
cơ bản `B0/P-P/P-C/P-PC` đã có trong `geometry_analysis`; runner ở đây chỉ train
các control còn thiếu. Mọi training run trong `run_controls.sh` bị khóa bằng:

```text
--selection-protocol test
```

Do đó IEMOCAP dùng toàn bộ 120 training conversations và chọn checkpoint bằng
test weighted F1, giống protocol SDT/CSS hiện tại.

## Các control

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| `P-0` | Có cùng Poincaré/Wheel branch nhưng hai lambda bằng 0 | Gain có thực sự đến từ loss? |
| `RAND-P` | Prototype loss với label-to-angle bị hoán vị | Prototype đúng về semantic có ích không? |
| `RAND-C` | CPCC với semantic distance bị hoán vị | Semantic ordering đúng có ích không? |
| `RAND-PC` | Cả hai loss với cùng random Wheel | Wheel đúng có tốt hơn auxiliary task ngẫu nhiên không? |
| `DETACH-PC` | Hai loss nhưng detach `H_TT/H_AA/H_VV` trước projector | Gradient Wheel vào backbone có cần thiết không? |
| `P-FIXED` | Hai loss, mọi sample có radius 0.75 | Radial freedom có đóng góp không? |

Random Wheel dùng generator riêng (`PERMUTATION_SEED=1729`) nên không tiêu thụ
global training RNG. Code loại bỏ rotation/reflection vì hai phép đó bảo toàn mọi
khoảng cách vòng tròn và không phá semantic.

## Chạy tất cả control

Một seed:

```bash
cd /home/ubuntu/26anh.vl/SDT_reliability
bash experiment/emotion_wheel_mechanism/run_all.sh --gpu-id 0
```

Năm seed:

```bash
SEEDS="2024 2025 2026 2027 2028" \
  bash experiment/emotion_wheel_mechanism/run_all.sh --gpu-id 0
```

Chạy riêng một control:

```bash
SEEDS="2024" bash experiment/emotion_wheel_mechanism/run_controls.sh RAND-PC --gpu-id 0
```

Có thể đặt output riêng:

```bash
OUTPUT_DIR="$PWD/experiment/emotion_wheel_mechanism/results/iemocap_5seeds" \
SEEDS="2024 2025 2026 2027 2028" \
  bash experiment/emotion_wheel_mechanism/run_all.sh --gpu-id 0
```

## Tổng hợp cùng các run đúng-Wheel cũ

Giả sử control mới nằm ở:

```text
experiment/emotion_wheel_mechanism/results/iemocap
```

và ablation cũ `B0/P-P/P-C/P-PC` nằm ở:

```text
geometry_analysis/results/iemocap
```

chạy:

```bash
DEVICE=cuda GPU_ID=0 \
  bash experiment/emotion_wheel_mechanism/run_analysis.sh \
  experiment/emotion_wheel_mechanism/results/iemocap \
  geometry_analysis/results/iemocap
```

Kết quả chính:

```text
mechanism_summary/mechanism_runs.csv
mechanism_summary/mechanism_summary.csv
mechanism_summary/mechanism_paired_deltas.csv
mechanism_summary/mechanism_summary.md
mechanism_summary/control_representation_probes.csv
```

Summarizer luôn tính semantic error bằng **Emotion Wheel đúng**, kể cả checkpoint
được train bằng random Wheel. Vì vậy `TRUE-PC` và `RAND-PC` được so sánh theo
cùng một thước đo semantic.

## Probe representation đúng-Wheel cũ

```bash
python experiment/emotion_wheel_mechanism/probe_representations.py \
  geometry_analysis/results/iemocap/<TRUE-PC-run>/best_checkpoint.pt \
  --device cuda --gpu-id 0
```

Probe train cùng một logistic-regression classifier trên `pure` và `enhanced`
features, rồi báo cáo accuracy, weighted/macro F1 và Spearman giữa khoảng cách
class-centroid với Emotion Wheel đúng.

## Gradient alignment

Chạy trên checkpoint `P-PC` đúng-Wheel:

```bash
python experiment/emotion_wheel_mechanism/gradient_alignment.py \
  geometry_analysis/results/iemocap/<TRUE-PC-run>/best_checkpoint.pt \
  --device cuda --gpu-id 0 --batches 20
```

Output `gradient_alignment.csv/json` chứa cosine giữa gradient SDT loss và tổng
hai Wheel loss trên ba pure encoders Text/Audio/Visual. Cosine dương cho thấy
hai objective cùng hướng cục bộ; cosine âm cho thấy gradient conflict.

## Cách kết luận

```text
TRUE-PC > P-0        : gain đến từ Wheel objective.
TRUE-PC > RAND-PC    : semantic ordering đúng có đóng góp.
TRUE-PC > DETACH-PC  : Wheel cải thiện backbone representation.
TRUE-PC > P-FIXED    : radial freedom có đóng góp.
semantic error/far error giảm: prediction tốt hơn theo đúng cấu trúc Wheel.
pure probe tăng trước enhanced probe: phù hợp chuỗi cơ chế pure → enhanced → fusion.
```

Vì checkpoint được chọn trên test theo yêu cầu, kết quả phù hợp để so sánh với
protocol SDT gốc. Khi viết claim generalization nghiêm ngặt, cần ghi rõ test đã
tham gia checkpoint selection.

