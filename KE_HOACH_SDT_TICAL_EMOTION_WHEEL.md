# Kế hoạch triển khai SDT + TiCAL + Emotion Wheel trong Poincaré ball

## 1. Mục tiêu

Mở rộng TiCAL hiện tại bằng một prior hình học có thể giải thích được:

- Mỗi emotion class có một prototype cố định trong mặt phẳng đầu tiên của
  Poincaré ball.
- Góc prototype tuân theo thứ tự semantic của emotional wheel.
- Bán kính prototype nằm chặt bên trong ball và giống nhau giữa các class ở
  thí nghiệm đầu tiên.
- Anchor bank vẫn mô hình hóa phân bố dữ liệu thật; prototype chỉ cung cấp
  cấu trúc class toàn cục.
- Các mode TiCAL cũ phải giữ nguyên tuyệt đối khi emotion wheel bị tắt.

## 2. Thứ tự class

IEMOCAP dùng label mapping của SDT:

```text
0 happy, 1 sad, 2 neutral, 3 angry, 4 excited, 5 frustrated
```

Thứ tự trên wheel:

```text
happy -> excited -> angry -> frustrated -> sad -> neutral
```

hay class order `(0, 4, 3, 5, 1, 2)`.

MELD dùng:

```text
0 neutral, 1 surprise, 2 fear, 3 sadness,
4 joy, 5 disgust, 6 anger
```

Thứ tự wheel được dùng là `(4, 1, 2, 6, 5, 3, 0)`.

## 3. Prototype hyperbolic

Với class `c` ở vị trí `k` trên wheel:

```text
theta_c = 2*pi*k/C
p_c = radius * [cos(theta_c), sin(theta_c), 0, ..., 0]
```

Prototype là buffer cố định, được lưu trong checkpoint nhưng không nhận
gradient. Điều kiện `0 < radius < 1 - hyp_eps` giữ prototype cách biên vô hạn
của Poincaré ball.

Prototype logits:

```text
logit_c(z) = -d_Poincare(z, p_c) / wheel_temperature
```

Prototype typicality:

```text
tau_proto = exp(-min_c d_Poincare(z, p_c) / wheel_temperature)
```

## 4. Kết hợp prototype và anchor

TiCAL cũ tạo `tau_anchor` từ khoảng cách nearest anchor. Khi wheel bật:

```text
tau = tau_anchor ** (1 - wheel_anchor_mix)
      * tau_proto ** wheel_anchor_mix
```

`wheel_anchor_mix=0` chỉ dùng anchor; `1` chỉ dùng prototype. Giá trị mặc định
cho thí nghiệm đầu tiên là `0.5`.

## 5. Wheel-aware modality disagreement

Giữ pseudo-label từ nearest anchor nhưng thay chỉ báo `khác/giống` bằng
circular class distance:

```text
delta(i,j) = acos(cos(theta_i - theta_j)) / pi
```

Disagreement của utterance là trung bình `TA`, `TV`, `AV`. Consistency vẫn dùng
công thức TiCAL:

```text
kappa = (tau_T * tau_A * tau_V) ** consistency_t
        * exp(-consistency_k * wheel_disagreement)
```

## 6. Hai loss mới có thể ablation độc lập

### 6.1 Prototype classification

```text
L_wheel_proto = mean_m CE(logits_proto_m, ground_truth)
```

Loss này kéo pure feature của từng modality về vùng emotion đúng.

### 6.2 Wheel CPCC

Thay categorical distance nhị phân trong một CPCC riêng bằng circular distance
giữa hai ground-truth class:

```text
L_wheel_cpcc = 1 - corr(
    pairwise Poincare feature distance,
    pairwise wheel class distance
)
```

Loss tổng:

```text
L = L_SDT/TiCAL
    + lambda_wheel_proto * L_wheel_proto
    + lambda_wheel_cpcc * L_wheel_cpcc
```

HypCPCC pseudo-label hiện có vẫn là một loss riêng và không bị thay đổi.

## 7. CLI và ablation

Các tùy chọn mới:

```text
--use-emotion-wheel
--wheel-prototype-radius
--wheel-temperature
--wheel-anchor-mix
--lambda-wheel-proto
--lambda-wheel-cpcc
```

Một bash riêng chạy cấu hình đầy đủ. Có thể đặt từng lambda bằng `0` hoặc đặt
`wheel_anchor_mix` bằng `0` để tách tác dụng của từng thành phần.

## 8. Diagnostics

Log và CSV phải phân biệt:

- `anchor_tau_*`: typicality từ anchor cũ.
- `prototype_tau_*`: typicality từ wheel prototype.
- `tau_*`: typicality sau khi kết hợp.
- `anchor_pseudo_label_*` và `wheel_pseudo_label_*`.
- categorical disagreement cũ và wheel disagreement mới.
- raw/weighted prototype loss và wheel CPCC.

## 9. Kiểm thử bắt buộc

1. Prototype nằm trong ball và đúng thứ tự góc.
2. Circular distance đối xứng, đường chéo bằng 0, class cạnh nhau gần hơn class
   đối diện.
3. Feature tại đúng prototype có prototype CE nhỏ và predicted class đúng.
4. Blended typicality đúng ở mix `0`, `0.5`, `1`.
5. Wheel disagreement phân biệt disagreement gần và xa.
6. Gradient từ hai wheel loss tới projector và SDT intra-modal encoder.
7. Tắt wheel cho kết quả y hệt TiCAL cũ.
8. Smoke train, checkpoint reload và test export hoàn tất.

