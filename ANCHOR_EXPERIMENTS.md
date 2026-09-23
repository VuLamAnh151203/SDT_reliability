# Thí nghiệm chất lượng anchor trên MELD

File `exec_meld_anchor_experiment.sh` chạy CA-KD + HypCPCC và không bật COLD hay
Emotion Wheel. Bốn mode chỉ khác nhau ở cách xây dựng và kích hoạt anchor bank.

```bash
# Kết quả equal cũ: teacher dùng chung cho ba bank, không đợi class coverage.
bash exec_meld_anchor_experiment.sh equal --gpu-id 0

# Equal + chỉ query khi mỗi lớp trong cả ba bank có ít nhất 8 anchor.
bash exec_meld_anchor_experiment.sh coverage --gpu-id 0

# Equal + từng bank chỉ nhận anchor khi classifier của modality đó dự đoán đúng.
bash exec_meld_anchor_experiment.sh modality --gpu-id 0

# Bật cả class coverage và modality-specific admission.
bash exec_meld_anchor_experiment.sh combined --gpu-id 0
```

Ngưỡng coverage mặc định là 8. Có thể đổi bằng biến môi trường:

```bash
ANCHOR_MIN_PER_CLASS=16 bash exec_meld_anchor_experiment.sh combined --gpu-id 0
```

Hoặc override trực tiếp ở cuối lệnh:

```bash
bash exec_meld_anchor_experiment.sh combined \
  --anchor-min-per-class 4 --gpu-id 0
```

Mặc định script dùng `--tical-mode hyp`, tương ứng CA-KD + HypCPCC. Ablation chỉ
CA-KD có thể chạy bằng:

```bash
TICAL_MODE=kd bash exec_meld_anchor_experiment.sh combined --gpu-id 0
```

Ý nghĩa hai tùy chọn mới:

- `--anchor-min-per-class 0`: giữ cơ chế readiness cũ, chỉ cần mỗi bank không
  rỗng. Giá trị lớn hơn 0 yêu cầu mọi lớp trong mọi bank đạt ngưỡng đó.
- `--anchor-admission teacher`: dùng fused teacher đúng và confidence lớn hơn
  threshold cho cả ba bank như trước.
- `--anchor-admission modality`: ngoài điều kiện teacher, bank T/A/V còn yêu cầu
  student classifier T/A/V tương ứng dự đoán đúng nhãn thật.

Trong mode `combined`, một lớp khó có thể làm `tical_ready_rate` giữ ở 0 lâu hơn.
Đây là hành vi mong đợi: SDT tiếp tục train và xây bank, còn CA-KD/HypCPCC chưa
được dùng cho đến khi bank đủ coverage. Theo dõi các cột
`anchor_{t,a,v}_class_*` và `tical_ready_rate` trong `epoch_metrics.csv`.
