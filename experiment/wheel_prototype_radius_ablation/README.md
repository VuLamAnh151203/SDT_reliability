# Ablation bán kính Emotion-Wheel prototype trong SDT_new

Bốn cấu hình đều giữ nguyên SDT, Poincaré 16D, sample radius tự do,
Prototype loss `0.1` và Wheel CPCC `0.05`. Chỉ cách đặt bán kính prototype
thay đổi:

| Mode | Số tham số radius | Regularization |
|---|---:|---:|
| `fixed-shared` | 0 | 0 |
| `learnable-shared` | 1 | 0 |
| `learnable-per-class` | số lớp | 0 |
| `per-class-regularized` | số lớp | `lambda=0.01` mặc định |

Radius học được dùng tham số hóa sigmoid trong `[0.20, 0.90]` và khởi tạo
đúng tại `0.75`. Mode cuối thêm:

\[
L_{radius}=\frac{1}{C}\sum_c(r_c-0.75)^2.
\]

Chạy cả bốn mode trên IEMOCAP, một seed và chọn checkpoint bằng test:

```bash
cd SDT_new
bash experiment/wheel_prototype_radius_ablation/run_ablation.sh all \
  --gpu-id 2
```

Chạy riêng một mode:

```bash
bash experiment/wheel_prototype_radius_ablation/run_ablation.sh \
  learnable-per-class --gpu-id 2
```

Chọn checkpoint bằng validation:

```bash
SELECTION_PROTOCOL=validation \
bash experiment/wheel_prototype_radius_ablation/run_ablation.sh all \
  --gpu-id 2
```

Chạy MELD:

```bash
DATASET=MELD \
bash experiment/wheel_prototype_radius_ablation/run_ablation.sh all \
  --gpu-id 2
```

Sau khi hoàn tất, script tự sinh `radius_ablation_runs.csv` và
`radius_ablation_aggregate.csv`. Cột `learned_radii` chứa bán kính theo thứ tự
class ID của dataset.
