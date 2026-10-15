# Sweep fixed Emotion-Wheel prototype radius

Thí nghiệm này giữ nguyên SDT + Poincaré 16D + cả hai Wheel loss và chỉ đổi
bán kính cố định dùng chung cho mọi class prototype:

```text
requested radius = 0.25, 0.50, 0.75, 1.00
```

Sample embedding vẫn dùng `--wheel-radius-mode free`. Vì Poincaré ball là miền
mở `0 <= ||p|| < 1`, trường hợp yêu cầu `1.00` được chạy bằng `0.9999`. Không
thể đặt prototype chính xác tại `1.0` vì khoảng cách hyperbolic tiến tới vô hạn.

Để radius `0.75` khớp với run tham chiếu TRUE-PC đạt `75.306` weighted F1,
runner mặc định dùng đúng:

```text
tical_warmup_epochs = 10
num_workers         = 0
num_threads         = 0
```

Không dùng các default tối ưu tốc độ của ablation learnable-radius, vì thay đổi
số DataLoader worker có thể làm thứ tự mini-batch khác đi dù vẫn dùng seed 2024.

Chạy IEMOCAP, seed 2024, chọn checkpoint trên test:

```bash
cd /home/ubuntu/26anh.vl/SDT_reliability

GPU_ID=0 SEEDS="2024" SELECTION_PROTOCOL=test \
bash experiment/wheel_prototype_radius_ablation/run_fixed_radius_sweep.sh \
  --gpu-id 0
```

Nếu chấp nhận kết quả không còn là reproduction chính xác để chạy nhanh hơn:

```bash
NUM_THREADS=1 NUM_WORKERS=2 WARMUP_EPOCHS=150 \
GPU_ID=0 SEEDS="2024" \
bash experiment/wheel_prototype_radius_ablation/run_fixed_radius_sweep.sh \
  --gpu-id 0
```

Đổi thư mục output:

```bash
OUTPUT_DIR="$PWD/experiment/wheel_prototype_radius_ablation/results/radius_sweep_seed2024" \
GPU_ID=0 SEEDS="2024" \
bash experiment/wheel_prototype_radius_ablation/run_fixed_radius_sweep.sh \
  --gpu-id 0
```

Chạy nhiều seed:

```bash
GPU_ID=0 SEEDS="2024 2025 2026 2027 2028" \
bash experiment/wheel_prototype_radius_ablation/run_fixed_radius_sweep.sh \
  --gpu-id 0
```

Sau khi hoàn tất, runner sinh:

```text
fixed_radius_sweep_runs.csv
fixed_radius_sweep_summary.md
```

Hai file chứa performance của từng radius và chênh lệch weighted F1 so với
radius `0.75` ở cùng seed.

Chạy riêng radius `0.75` để kiểm tra reproduction:

```bash
OUTPUT_DIR="$PWD/experiment/wheel_prototype_radius_ablation/results/check_0p75" \
RADII="0.75" GPU_ID=0 SEEDS="2024" \
bash experiment/wheel_prototype_radius_ablation/run_fixed_radius_sweep.sh \
  --gpu-id 0
```
