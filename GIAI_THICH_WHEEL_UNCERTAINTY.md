# SDT + Poincaré Wheel + uncertainty-aware fusion

Mode này giữ nguyên SDT và hai loss đang có:

\[
\mathcal L_{base}=\mathcal L_{SDT}
+\lambda_{proto}\mathcal L_{WheelProto}
+\lambda_{cpcc}\mathcal L_{WheelCPCC}.
\]

Mỗi Poincaré embedding được đưa về tangent space tại gốc. Một MLP riêng cho
Text, Audio và Visual dự đoán scalar uncertainty trong `[0,1]`:

\[
\hat u_i^m=\operatorname{sigmoid}(MLP_m(\log_0(z_i^m))).
\]

Target trong training là expected emotion-wheel error:

\[
t_i^m=\sum_c q_i^m(c)D_{wheel}(c,y_i),
\qquad
q_i^m(c)=\operatorname{softmax}_c(-d_{\mathbb H}(z_i^m,p_c)/T).
\]

Target được stop-gradient và uncertainty loss là MSE. Ground-truth không được
dùng trong forward hoặc inference.

Fusion confidence được chuẩn hóa theo chiều modality:

\[
w_i^m=\operatorname{softmax}_m(-\hat u_i^m/\tau_u).
\]

Weight này điều chỉnh và chuẩn hóa lại SDT per-feature gate. Student
classifiers vẫn dùng enhanced feature gốc. Last layer của ba uncertainty head
được khởi tạo bằng zero, vì vậy ban đầu `u=0.5`, `w=1/3` và fusion khớp SDT
baseline.

## Các mode

| Mode | Uncertainty loss | Adaptive fusion |
|---|---:|---:|
| `none` | Không | Không |
| `gate` | Không | Có |
| `supervise` | Có | Không |
| `full` | Có | Có |

## Chạy MELD

Full mode:

```bash
bash exec_meld_wheel_uncertainty.sh full --gpu-id 0
```

Vì `full` là mặc định, có thể chạy ngắn hơn:

```bash
bash exec_meld_wheel_uncertainty.sh --gpu-id 0
```

Chạy từng ablation:

```bash
bash exec_meld_wheel_uncertainty_ablation.sh none --gpu-id 0
bash exec_meld_wheel_uncertainty_ablation.sh gate --gpu-id 0
bash exec_meld_wheel_uncertainty_ablation.sh supervise --gpu-id 0
bash exec_meld_wheel_uncertainty_ablation.sh full --gpu-id 0
```

Chạy liên tiếp cả bốn mode:

```bash
bash exec_meld_wheel_uncertainty_ablation.sh all --gpu-id 0
```

Các argument đặt cuối lệnh override default. Ví dụ:

```bash
bash exec_meld_wheel_uncertainty.sh full \
  --wheel-uncertainty-gate-strength 1.0 \
  --wheel-uncertainty-temperature 0.5 \
  --gpu-id 0
```

## Default của full mode

```text
lambda Wheel Prototype       0.1
lambda Wheel CPCC            0.05
lambda uncertainty           0.1
uncertainty hidden dim       32
uncertainty temperature      1.0
gate strength                0.5
warm-up                      10 epochs
ramp                         5 epochs
```

Trong warm-up, uncertainty loss và uncertainty gate đều có hệ số zero. Sau
warm-up, cả hai tăng tuyến tính đến giá trị đầy đủ trong `ramp` epochs.

`epoch_metrics.csv` ghi prediction/target/weight mean theo modality, target và
prediction standard deviation, MAE, Pearson correlation, uncertainty trung
bình của case đúng/sai, cùng entropy của modality weights. Target standard
deviation gần zero là dấu hiệu target collapse.
