# SDT_new: SDT + COLD + TiCAL

## Chạy SDT + TiCAL (không dùng COLD)

Script riêng:

```bash
bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0
```

Mặc định script chạy **SDT + TiCAL consistency-aware KD** (`tical-mode=kd`).
Nó luôn dùng `fusion-variant=sdt`, không tạo Gaussian distribution head, không
sampling, không đọc OOF reliability, đồng thời đặt `lambda-co=0`, `lambda-reg=0`
và `lambda-reliability=0`. Code cũng báo lỗi nếu cố bật TiCAL cùng `guided`,
`replace` hoặc `oof-guided`.

Các ablation theo guideline:

```bash
# E1: chỉ đo và log TiCAL, loss/prediction vẫn là SDT
TICAL_MODE=observe bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0

# E2 (mặc định): consistency-aware self-distillation
TICAL_MODE=kd bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0

# E3: E2 + HypCPCC
TICAL_MODE=hyp bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0

# E4: E2 + consistency-aware fusion
TICAL_MODE=fusion bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0

# E5: E2 + HypCPCC + consistency-aware fusion
TICAL_MODE=full bash SDT_new/exec_iemocap_tical.sh --device cuda --gpu-id 0
```

TiCAL lấy `H_tt`, `H_aa`, `H_vv` ngay sau ba intra-modal Transformer và trước
unimodal gate/cross-modal fusion. Ba `HyperbolicProjector` đưa feature vào
Poincare ball. Mỗi modality có một FIFO HASL riêng. Trong train, batch hiện tại
luôn query bank cũ trong forward; chỉ sau `optimizer.step()` mới thêm các mẫu mà
fused teacher dự đoán đúng và có confidence lớn hơn `anchor-conf-threshold`.
Validation/test chỉ query bank đã học từ train và tuyệt đối không update bank.

Warm-up mặc định là 5 epoch. Epoch 1--5 dùng đúng loss SDT và chỉ xây HASL;
TiCAL bắt đầu query từ epoch 6. Các thiết lập chính có thể override ở cuối lệnh:

```bash
bash SDT_new/exec_iemocap_tical.sh \
  --tical-warmup-epochs 5 \
  --anchor-size 2048 --anchor-conf-threshold 0.8 \
  --hyperbolic-dim 128 \
  --consistency-t 0.2 --consistency-k 0.5
```

Mỗi epoch ghi đầy đủ vào `epoch_metrics.csv`: kích thước và số anchor mỗi class,
agreement T/A/T/V/A/V/T=A=V, mean/std/q05/q50/q95 của từng `tau`, phân phối
`kappa`, tỷ lệ ba nhóm consistency, accuracy từng nhóm, original KL,
consistency-aware KL và HypCPCC. `test_predictions.csv` có thêm pseudo-label,
`tau`, `label_discrepancy` và `kappa` cho từng utterance.

Script vẫn giữ protocol hiện tại của dự án: dùng test F1 để chọn checkpoint.
Muốn tách validation từ train, truyền `--selection-protocol validation`.

Triển khai theo hướng dẫn được cung cấp: giữ encoder SDT, thêm Gaussian distribution
head sau mỗi enhanced representation, dùng lại ba student classifier, và đưa
reliability vào multimodal fusion. Variant B (`guided`) là mặc định.

Code encoder/classifier được tách từ `../SDT/model.py`, nhánh
`appraisal_mode='none'`. `ORIGIN.txt` ghi SHA256 của file nguồn tại thời điểm tách.
Thư mục này tự chứa code cần chạy; dữ liệu có thể dùng chung với `../SDT/data/`.

## Những phần giữ nguyên

- Temporal convolution cho text/audio/visual; position và speaker embeddings.
- Chín intra/inter-modal transformer và chín unimodal gate.
- Ba phép giảm chiều tạo `H'_T`, `H'_A`, `H'_V`.
- Ba student classifier `ReLU → Dropout → Linear` và teacher classifier `Linear`.
- CE teacher, tổng CE student và tổng KL self-distillation với temperature.

`sdt_backbone.py` chứa các thành phần này. `model.py` bổ sung COLD sau `H'`.
Không cần appraisal, speech-concept, CSE hoặc spherical router để chạy.

## Distribution, score và hai biến thể fusion

Mỗi modality có hai Linear độc lập:

```text
mu_m       = Linear_mu(H'_m)
logvar_m   = clamp(Linear_logvar(H'_m), logvar_min, logvar_max)
variance_m = exp(logvar_m)

train: z_m = mu_m + exp(0.5 * logvar_m) * epsilon, epsilon ~ N(0,I)
eval:  z_m = mu_m

student_logits_m = student_m(z_m)
v_m = ||variance_m||_2
s_m = 1 / (v_m + eps)
reliability_logit_m = -log(v_m + eps)
r_m = s_m / (s_T + s_A + s_V)
```

Distribution head có hai chế độ khởi tạo:

| `--distribution-init` | Khởi tạo |
| --- | --- |
| `random` — mặc định cũ | Hai Linear dùng khởi tạo mặc định của PyTorch |
| `sdt-preserving` | `W_mu=I`, `b_mu=0`, `W_logvar=0`, `b_logvar=initial_logvar` |

Với `sdt-preserving --initial-logvar -6`, tại thời điểm khởi tạo:

```text
mu_m = H'_m
variance_m = exp(-6) ≈ 0.00248
sampling std = exp(-3) ≈ 0.0498
r_T = r_A = r_V = 1/3
guided fusion = SDT learned-gate fusion (ở eval)
```

Các head vẫn train bình thường sau initialization. `initial_logvar` phải nằm
trong khoảng `logvar_min..logvar_max`.

`H'`, `mu`, `logvar`, `z` có shape `[batch, sequence, hidden_dim]`.
`r` có shape `[batch, sequence, 3]`, thứ tự **T, A, V**.

| `--fusion-variant` | Công thức |
| --- | --- |
| `replace` — A | `h = sum_m r_m * z_m`; không dùng learned multimodal gate |
| `guided` — B, mặc định | `g = softmax_m(W * H'_m)`, `g_bar = normalize_m(g * r)`, `h = sum_m g_bar_m * z_m` |
| `oof-guided` | Học reliability từ OOF target, giữ classifier/fusion trên `H'`, không Gaussian bottleneck hay sampling |
| `sdt` — đối chứng | SDT deterministic: classifier và learned gate nhận `H'`; không tạo distribution head, COLD/reg bằng 0 |

Gate trong SDT nguồn có trọng số **theo từng chiều feature**, shape
`[batch, sequence, 3, hidden_dim]`. B giữ đúng cấu trúc này và broadcast scalar
`r_m` trên hidden dimension. Code dùng `softmax(W*H' + log(r))` để tính `g_bar`
ổn định hơn, tương đương phép nhân rồi chuẩn hóa trong hướng dẫn.

Quan hệ trong implementation được chỉnh theo cùng một chiều ngữ nghĩa:
`prediction error cao → variance cao → reliability thấp → fusion weight thấp`.
COLD ghép prediction quality `-CE` với reliability logit `-log(v + eps)`.
Softmax của reliability logit tạo xác suất tỷ lệ với `1/(v + eps)`, nhưng ổn
định hơn việc đưa trực tiếp reciprocal vào softmax.

## Loss và những lựa chọn cần ghi lại khi làm thí nghiệm

Chỉ các utterance có `umask > 0` tham gia loss. Padding bị loại **trước** khi
tính CE, softmax COLD và Gaussian regularizer.

```text
D_m = cross_entropy(student_logits_m[valid], labels[valid], reduction='none')
Q_m = -D_m
R_m = -log(v_m + eps)
symKL(x,y) = KL(softmax(x) || softmax(y)) + KL(softmax(y) || softmax(x))

L_CO_m   = symKL(Q_m, R_m)
L_CO_TAV = symKL(cat(Q_T,Q_A,Q_V), cat(R_T,R_A,R_V))
L_COLD   = L_CO_T + L_CO_A + L_CO_V + L_CO_TAV

L_SDT = gamma_1 * CE_teacher
      + gamma_2 * (CE_T + CE_A + CE_V)
      + gamma_3 * (KL_T + KL_A + KL_V)

L_total = L_SDT + lambda_co * L_COLD + lambda_reg * L_reg
```

Softmax COLD chạy trên vector **tất cả utterance hợp lệ trong minibatch** của
mỗi modality. Thành phần TAV chạy trên một vector chiều `3 * N_valid`, theo
đúng phép concatenate được cung cấp. Không softmax theo class hay chỉ ba
modality của từng utterance. KL dùng tổng trên support, không chia thêm cho
`N_valid`. Loss COLD do đó phụ thuộc cách chia minibatch; cần giữ batch size và
protocol giống nhau khi so sánh.

Các chi tiết đề xuất chưa chỉ rõ được triển khai như sau:

1. **`L_reg`:** chọn Gaussian prior KL, vì hướng dẫn chỉ nêu tên regularizer:
   `L_reg_m = mean_valid,hidden[0.5 * (mu² + exp(logvar) - 1 - logvar)]`.
   `L_reg` là tổng T/A/V. Đây là giả định triển khai, không khẳng định là
   regularizer của COLD gốc. KL được lấy trung bình cả utterance lẫn latent
   dimension để giá trị không tăng tỷ lệ với `hidden_dim`. Dùng
   `--lambda-reg 0` để tắt.
2. **CE target của COLD:** mặc định `D_m.detach()` để COLD điều chỉnh variance
   theo prediction error; student vẫn học qua CE/KL SDT. Dùng
   `--no-detach-errors` nếu muốn gradient COLD chạy qua cả CE target.
3. **Chặn logvar:** mặc định `[-8, 8]`; thay bằng `--logvar-min/--logvar-max`.
   `eps=1e-8`. Variance norm và reciprocal được tính bằng float32.
4. **Trọng số:** `gamma_1=gamma_2=gamma_3=1`, `lambda_co=0.1`,
   `lambda_reg=0.1` là cấu hình khởi đầu, chưa được tuning.

IEMOCAP giữ class weights của `SDT/train.py` cho các CE của SDT. CE dùng làm
target COLD luôn **không có class weight** để phản ánh error từng utterance
theo công thức bạn gửi. `--no-class-weight` tắt class weights của SDT.

Self-distillation giữ hành vi code SDT nguồn: teacher probability **không
detach**, và KL **không nhân thêm temperature²**. Việc detach CE target COLD
không thay đổi gradient này.

## OOF reliability và pruning

`oof-guided` là pipeline riêng để reliability không được tạo bởi model đã nhìn
thấy chính training sample đó:

```text
trainVid --K-fold theo dialogue--> K baseline SDT
       --predict holdout--> CE_T, CE_A, CE_V
       --softmax(-CE / tau)--> OOF reliability target
       --train reliability heads--> predicted reliability
       --guided SDT gate--> emotion prediction
```

Mỗi dialogue chỉ xuất hiện trong holdout của đúng một fold. Các fold OOF không
dùng `testVid` và train đúng số epoch cố định, không chọn epoch bằng holdout.
Model cuối vẫn mặc định chọn checkpoint bằng test giống SDT theo yêu cầu hiện
tại.

Tạo OOF targets (5 baseline folds, 50 epoch/fold):

```bash
bash SDT_new/exec_iemocap_build_oof_reliability.sh \
  --device cuda --gpu-id 0
```

File mặc định là
`SDT_new/reliability_targets/iemocap_oof_seed2024.csv`; JSON cùng tên lưu cấu
hình, fold IDs và xác nhận test không tham gia. Nếu muốn baseline OOF hội tụ kỹ
hơn có thể ghi đè budget khi tạo file mới:

```bash
OOF_TARGETS=SDT_new/reliability_targets/iemocap_oof_150ep_seed2024.csv \
bash SDT_new/exec_iemocap_build_oof_reliability.sh \
  --device cuda --gpu-id 0 --epochs 150
```

Train model cuối:

```bash
bash SDT_new/exec_iemocap_oof_guided.sh \
  --device cuda --gpu-id 0
```

Nếu dùng target path khác, đặt cùng biến môi trường khi train:

```bash
OOF_TARGETS=SDT_new/reliability_targets/iemocap_oof_150ep_seed2024.csv \
bash SDT_new/exec_iemocap_oof_guided.sh --device cuda --gpu-id 0
```

Mode này khởi tạo ba reliability head bằng 0, nên ban đầu reliability là
`[1/3,1/3,1/3]` và guided fusion bằng đúng learned gate của SDT. Student và
teacher luôn nhận `H'`; không có `mu`, `z`, Gaussian REG hay sampling noise.
Loss bổ sung là:

```text
L_reliability = mean_valid KL(r_OOF || softmax(reliability_logits))
L_total = L_SDT + lambda_reliability * L_reliability
```

Script mặc định dùng:

```text
lambda_reliability       = 1.0
modality_prune_quantile  = 0.10
sample_prune_quantile    = 0.05
disagreement_weight      = 1.0
```

Modality pruning bỏ 10% reliability thấp nhất của từng modality khỏi CE/KD
student. Modality tốt nhất của mỗi utterance luôn được giữ. Sample pruning tạo
noise score `mean(CE_T,CE_A,CE_V) + disagreement_weight * JSD`, lấy threshold
riêng cho từng emotion class để giữ cân bằng lớp, và mặc định chỉ prune khi cả
ba modality đều dự đoán sai. `--sample-prune-any` bỏ điều kiện bảo thủ này.

Pruning không xóa utterance khỏi dialogue: utterance vẫn đi qua encoder để giữ
conversational context, nhưng loss classification/reliability của sample bị
mask. Tại inference không cần OOF file; reliability head dự đoán trực tiếp từ
`H'`. Console log `reliability`, `sample_keep_rate` và `modality_keep_rate`.

Các ablation quan trọng:

```bash
# Chỉ OOF soft-guided gate, không prune
bash SDT_new/exec_iemocap_oof_guided.sh \
  --modality-prune-quantile 0 --sample-prune-quantile 0

# Chỉ modality pruning
bash SDT_new/exec_iemocap_oof_guided.sh \
  --modality-prune-quantile 0.10 --sample-prune-quantile 0

# Không đưa disagreement vào whole-sample noise score
bash SDT_new/exec_iemocap_oof_guided.sh --disagreement-weight 0
```

## Cài đặt và dữ liệu

```bash
cd SDT_new
python -m pip install -r requirements.txt
```

Chương trình lần lượt tìm pickle trong `SDT_new/data/` rồi `../SDT/data/`, hoặc
dùng đường dẫn chỉ định bởi `--feature-path`. Đường dẫn tương đối chỉ định qua
CLI được tính từ thư mục hiện tại. Code đọc schema SDT IEMOCAP 12 phần tử và
MELD 13 phần tử; dùng `videoText`, `videoAudio`, `videoVisual` như SDT nguồn.
Kích thước input được suy ra từ pickle.

## Chạy

Từ thư mục gốc workspace:

```bash
# IEMOCAP, Variant B (mặc định)
bash SDT_new/exec_iemocap.sh

# Variant B với initialization gần SDT, lambda_co=0.5, lambda_reg=0.01
bash SDT_new/exec_iemocap_guided_sdt_init.sh

# IEMOCAP, Variant A
bash SDT_new/exec_iemocap_replace.sh

# SDT deterministic để đối chứng cùng training harness
bash SDT_new/exec_original_sdt.sh

# MELD, B; thêm --fusion-variant replace để chạy A
bash SDT_new/exec_meld.sh
```

Các script nhận thêm tham số ở cuối, ví dụ:

```bash
bash SDT_new/exec_iemocap.sh --device cuda --gpu-id 0 --seed 2025 --lambda-co 0.05
```

Lệnh tương đương cho mode SDT-preserving:

```bash
bash SDT_new/exec_iemocap.sh \
  --fusion-variant guided \
  --distribution-init sdt-preserving \
  --initial-logvar -6 \
  --lambda-co 0.5 \
  --lambda-reg 0.01
```

Trên PowerShell có thể chạy Python trực tiếp:

```powershell
python SDT_new/train.py --Dataset IEMOCAP --fusion-variant guided --epochs 150
python SDT_new/train.py --Dataset IEMOCAP --fusion-variant replace --epochs 150
```

Trong môi trường Windows đã kiểm tra, lệnh `python` đang trỏ tới shim pyenv
chưa chọn version. Python có sẵn dùng để kiểm tra là 3.11.8; có thể gọi trực tiếp:

```powershell
& "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.11.8\python.exe" SDT_new/train.py --device cpu
```

Python này đang dùng PyTorch CPU. Khi train trên GPU, dùng môi trường Python
có bản PyTorch CUDA tương ứng và truyền `--device cuda`.

Script chạy foreground để hiển thị log và lỗi. Mỗi lần chạy tạo thư mục kết
quả riêng theo dataset, variant, seed và timestamp trong `SDT_new/results/`.
Biến môi trường `PYTHON` trong script Bash cho phép chọn Python executable.

## Protocol chọn checkpoint

Mặc định `--selection-protocol test` khớp membership và cách chọn epoch của
`SDT/train.py`: dùng toàn bộ `trainVid`, không có validation, chọn checkpoint
có weighted F1 cao nhất trên `testVid`. Do test tham gia chọn epoch, kết quả
này là **test-selected**, không phải đánh giá trên test độc lập.

Để chọn bằng validation và chỉ đánh giá test sau khi chọn xong:

```bash
bash SDT_new/exec_iemocap.sh --selection-protocol validation --valid-ratio 0.1
```

Validation lấy 10% dialogue đầu của `trainVid`, giống cách chia sampler SDT
khi `valid > 0`; train/validation/test không chồng lặp. Dùng cùng protocol,
seed và hyperparameter nền cho A, B và đối chứng. Các script chạy một seed
mỗi lần; có thể gọi lại với `--seed` khác.

## Checkpoint, log và inference

Mỗi run lưu:

- `config.json`, `split_ids.json`: cấu hình, đường dẫn dữ liệu, danh sách split.
- `epoch_metrics.csv`: từng thành phần SDT, COLD T/A/V/TAV, regularizer, Acc/F1.
- Console và CSV ghi cả COLD/reg thô lẫn `weighted_cold`/`weighted_reg`
  thực sự được cộng vào total loss.
- `best_checkpoint.pt`: model state, model config và metadata chọn epoch.
- `test_metrics.json`, `summary.json`, `classification_report.json`,
  `confusion_matrix.json`.
- `test_predictions.csv`: dialogue/index, label, prediction/probability,
  prediction student, CE error, variance norm, reliability, gate trung bình
  theo hidden dimension. Dòng padding không được xuất.

Đánh giá lại checkpoint:

```bash
python SDT_new/train.py --eval-checkpoint path/to/best_checkpoint.pt --device cpu
```

Architecture và loss settings được lấy từ checkpoint. Có thể chỉ định
`--feature-path` mới khi chuyển máy; schema và dialogue split phải khớp.
Checkpoint dùng để inference/evaluation, chưa cung cấp resume optimizer.

Model `forward` chỉ nhận features/masks/speakers/lengths, **không nhận nhãn**.
`model.eval()` dùng `z=mu`; không cần CE target hoặc COLD loss khi dự đoán:

```python
# Chạy trong SDT_new, hoặc thêm thư mục này vào sys.path.
import torch
from model import Transformer_Based_Model

checkpoint = torch.load("path/to/best_checkpoint.pt", map_location="cpu", weights_only=True)
model = Transformer_Based_Model(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    # text/visual/audio: [L,B,D_m], mask: [B,L], speakers: [B,L,S]
    output = model(text, visual, audio, mask, speakers, lengths)
    predictions = output["logits"].argmax(dim=-1)
    valid_predictions = predictions[mask.bool()]
```

## Kiểm tra

```bash
python -m unittest discover -s SDT_new/tests -v
python SDT_new/train.py --device cpu --epochs 1 --batch-size 2 \
  --hidden-dim 16 --n-head 2 --num-threads 1 --max-batches 1
```

Tests kiểm tra công thức KL/fusion, padding, gradient, sampling train/eval,
Gaussian regularizer, MELD speaker handling và đối chiếu encoder/baseline
với SDT nguồn khi file đó có mặt. `--max-batches` chỉ dùng kiểm tra luồng chạy;
kết quả được đánh dấu `smoke_test` và không dùng làm kết quả nghiên cứu.

## Nguồn SDT

Giữ attribution từ README nguồn:

```bibtex
@article{ma2024sdt,
  author={Ma, Hui and Wang, Jian and Lin, Hongfei and Zhang, Bo and Zhang, Yijia and Xu, Bo},
  journal={IEEE Transactions on Multimedia},
  title={A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations},
  year={2024},
  volume={26},
  pages={776-788},
  doi={10.1109/TMM.2023.3271019}
}
```
