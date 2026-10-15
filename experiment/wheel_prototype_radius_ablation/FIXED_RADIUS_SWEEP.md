# Sweep fixed Emotion-Wheel prototype radius

Thí nghiệm này giữ nguyên SDT + Poincaré 16D + cả hai Wheel loss và chỉ đổi
bán kính cố định dùng chung cho mọi class prototype:

```text
requested radius = 0.25, 0.50, 0.75, 1.00
```

Sample embedding vẫn dùng `--wheel-radius-mode free`. Vì Poincaré ball là miền
mở `0 <= ||p|| < 1`, trường hợp yêu cầu `1.00` được chạy bằng `0.9999`. Không
thể đặt prototype chính xác tại `1.0` vì khoảng cách hyperbolic tiến tới vô hạn.

Chạy IEMOCAP, seed 2024, chọn checkpoint trên test:

```bash
cd /home/ubuntu/26anh.vl/SDT_reliability

GPU_ID=0 SEEDS="2024" SELECTION_PROTOCOL=test \
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
