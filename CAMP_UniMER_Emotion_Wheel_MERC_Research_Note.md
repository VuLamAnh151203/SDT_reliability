# CAMP + UniMER cho Emotion-Wheel-Guided Multimodal Emotion Recognition in Conversation (MERC)

> **Research note / design proposal**  
> **Ngày:** 11/11/2026  
> **Phạm vi:** Phân tích hai paper, xác định thành phần có thể kế thừa và đề xuất các thí nghiệm cho hướng *emotion-wheel-guided feature representation*.  
> **Quy ước:** **[PAPER]** là phương pháp/nhận định được mô tả trong bài báo; **[PROPOSAL]** là giả thuyết, công thức hoặc thiết kế suy rộng cho MERC, **không phải** công thức nguyên bản của tác giả.

## 1. Research goal và kết luận nhanh

**Mục tiêu:** Học multimodal emotion representations vừa (i) phân biệt tốt các emotion classes, (ii) phản ánh quan hệ ngữ nghĩa/tâm lý giữa emotions, vừa (iii) giữ được những bằng chứng bổ sung, đôi khi bất đồng, từ text, audio và visual.

Hai bài bổ sung nhau ở hai cấp độ khác nhau:

| Khía cạnh | CAMP — ICML 2026 | UniMER — EMNLP 2026 |
|---|---|---|
| Câu hỏi chính | Làm sao học từ các modalities cung cấp bằng chứng bổ sung, tránh ép alignment không phù hợp? | Làm sao mô hình chú ý đến thông tin dành riêng cho từng emotion, nhất là minority classes? |
| Vai trò prototype | Đại diện bằng chứng/khái niệm ở cấp class, hỗ trợ alignment và fusion đa phương thức | Learnable class-specific **attention queries** để pooling contextual token features |
| Kỹ thuật đáng học | Class-wise optimal transport (OT), geometric constraints, tránh modality dominance | Class-prototype attention pooling; imbalance-aware training |
| Chưa giải quyết trực tiếp | Psychological emotion-wheel relations | Psychological geometry; liên kết từ attention queries tới semantic class centers |
| Cách dùng đề xuất | Căn chỉnh **shared semantic evidence**, bảo toàn **private modality evidence** | Trích xuất **emotion-specific features** trước khi fusion |

**Thông điệp thiết kế:** Không nên đơn giản nối hai mô hình rồi cộng tất cả losses. Nên tách ba vai trò: **emotion-specific attention** (UniMER-inspired), **cross-modal prototype correspondence** (CAMP-inspired), và **psychological semantic anchors** (emotion wheel).

---

## 2. UniMER: Phương pháp gốc và ý nghĩa cho MERC

**Paper:** *UniMER: Multilingual Multimodal Emotion Recognition with Prototype-Guided Attention Pooling* (EMNLP 2026)  
**URL:** https://aclanthology.org/2026.emnlp-main.1611/

### 2.1. Problem

**[PAPER]** Khi một utterance được gộp thành một vector global, tín hiệu quan trọng cho những emotion classes ít gặp có thể bị các pattern phổ biến lấn át. UniMER khai thác class-specific attention pooling để biểu diễn bằng chứng liên quan đến từng emotion.

Lưu ý: pipeline gốc dùng multimodal model tạo natural-language descriptions cho audio/visual/facial cues, sau đó đưa vào language backbone (LoRA). Đây **không** phải pipeline trực tiếp căn chỉnh các raw audio/visual embeddings bằng optimal transport.

### 2.2. Class-prototype attention pooling

Cho contextual token features (đã có dialogue/multimodal context):

\[
H=[h_1,\ldots,h_T],\quad h_t\in\mathbb R^d.
\]

Với \(K\) emotion classes, học \(K\) attention prototypes:

\[
P_{\mathrm{query}}=\{p_1,\ldots,p_K\}.
\]

**[PAPER]** Với emotion \(k\), tính attention score và attention weights:

\[
 e_t^{(k)}=p_k^\top\tanh(W_hh_t+b_h),\qquad
 \alpha_t^{(k)}=\frac{\exp(e_t^{(k)})}{\sum_{j=1}^T\exp(e_j^{(k)})}.
\]

Emotion-specific contextual representation:

\[
 c_k=\sum_{t=1}^T\alpha_t^{(k)}h_t.
\]

Các \(c_k\) được ghép và đưa qua prediction head:

\[
 s=\operatorname{MLP}([c_1;\ldots;c_K]),\qquad
 \hat y=\operatorname{softmax}(W_cs+b_c).
\]

Training gồm class-imbalance-aware cross entropy và regularization được paper mô tả (kèm label smoothing).

**Quan trọng:** \(p_k\) là **query để tìm thông tin**, không mặc nhiên là **centroid của emotion \(k\)** trong embedding space. Khoảng cách \(\operatorname{cos}(p_i,p_j)\) vì thế **không tự động** là khoảng cách semantic giữa hai emotion labels.

### 2.3. Điều có thể kế thừa

1. **[PROPOSAL]** Áp dụng query-per-emotion cho từng modality: \(p_{e}^{m}\), với \(m\in\{t,a,v\}\).
2. **[PROPOSAL]** Dùng queries để trích xuất \(c_{u,e}^{m}=\operatorname{Attn}(p_e^m,H_u^m)\) cho utterance \(u\).
3. **[PROPOSAL]** Thêm class-conditioned evidence heads để grounding các queries vào nhãn thay vì chỉ concatenation tất cả \(c_e\) qua một MLP chung.
4. **[PROPOSAL]** Tạo **semantic anchors riêng** \(q_e\) cho emotion wheel. Không ép attention queries \(p_e\) phải mang hình học của emotion wheel khi chưa có kiểm chứng.

### 2.4. Limitation đáng nghiên cứu

Một query được gán tên “Fear” không đảm bảo nó thực sự mã hóa Fear theo nghĩa semantic. Nếu output \([c_1;\ldots;c_K]\) được học cùng một MLP linh hoạt, nhiệm vụ có thể hoàn thành mà query không có một semantic role ổn định. Đây là **giả thuyết cần đo**, không phải lỗi đã được paper chứng minh.

**Diagnostic:** Attention specialization theo class, query similarity, per-class attribution, và intervention (mask một query để xem class nào bị ảnh hưởng).

---

## 3. CAMP: Phương pháp gốc và ý nghĩa cho MERC

**Paper:** *CAMP: Coherent Alignment of Multimodal Prototypes for Explainable Complementary Learning* (ICML 2026)  
**URL:** https://proceedings.mlr.press/v306/pellicer26a.html

### 3.1. Problem: complementary ≠ redundant evidence

**[PAPER]** Nhiều phương pháp alignment có xu hướng kéo các representations từ những modalities khác nhau lại gần nhau. Nhưng đối với complementary multimodal data, mỗi modality chỉ có thể mang một phần thông tin hoặc thậm chí một dấu hiệu có vẻ mâu thuẫn với modality còn lại.

**Ví dụ giả lập MERC:**

| Modal | Evidence | Emotion cue khả dĩ |
|---|---|---|
| Text | “I'm fine.” | Neutral |
| Audio | Giọng run, ngập ngừng | Fear |
| Visual | Nét mặt căng thẳng | Fear/Sadness |
| Context | Vừa xảy ra xung đột | Fear/Anger |

Loss kiểu \(\|z^t-z^a\|_2^2\) trên **toàn bộ** representation có thể làm mất modality-specific evidence. Tuy nhiên, không được suy ra rằng mọi cross-modal alignment đều có hại: cần kiểm chứng trên backbone/dataset cụ thể.

### 3.2. Key components từ CAMP

**[PAPER]** CAMP sử dụng class-wise optimal transport, prototype-based multimodal reasoning và các geometric constraints nhằm giữ coherent class evidence đồng thời hạn chế modality dominance / prototype collapse.

**Phạm vi xác nhận:** Các thành phần trên xuất hiện trong mô tả paper. Các bước, ký hiệu và objective ngay dưới đây được viết thành **[PROPOSAL] CAMP-inspired implementation for MERC**, không được xem là các equation được trích nguyên văn từ CAMP.

### 3.3. [PROPOSAL] Modality-specific emotion prototype banks

Cho encoded features:

\[
H_u^m=E_m(x_u^m),\qquad m\in\{t,a,v\}.
\]

Một class \(e\) có \(J\) prototypes tại modality \(m\):

\[
P_e^m=\{p_{e,1}^m,\ldots,p_{e,J}^m\}.
\]

Điều này cho phép, ví dụ, một emotion Fear có các acoustic modes khác nhau mà không cần giả định chúng là cùng một vector.

Nếu sử dụng 1 prototype/class/modality, việc áp dụng OT với transport matrix \(1\times1\) thường không mang lại matching đa dạng đáng kể. **Vì vậy, chỉ dùng OT nếu mỗi class có nhiều prototypes hoặc một phân phối evidence đủ phong phú.**

### 3.4. [PROPOSAL] Class-wise OT matching

Giả sử đã chiếu prototypes hai modality vào một shared semantic space và chuẩn hóa. Với class \(e\):

\[
 C_{ij}^{e,(a,v)}=1-\operatorname{cos}(p_{e,i}^{a},p_{e,j}^{v}).
\]

Tính transport plan:

\[
T_e^*=\arg\min_{T\in\Pi(\mu_e^a,\mu_e^v)}
\langle T,C_e\rangle+\varepsilon\sum_{ij}T_{ij}(\log T_{ij}-1).
\]

Trong đó \(\mu_e^a,\mu_e^v\) là mass/activation distributions, \(\varepsilon\) là entropy regularizer; Sinkhorn là một lựa chọn triển khai.

OT hỗ trợ **many-to-many matching** giữa prototype patterns, thay vì buộc prototype index \(j\) của audio phải tương ứng với index \(j\) của visual.

**Cảnh báo:** OT nhỏ không tự đảm bảo giữ complementarity; nếu ép alignment toàn bộ embeddings quá mạnh, model vẫn có thể làm mất evidence riêng từng modality.

### 3.5. [PROPOSAL] Shared/private decomposition

Để hạn chế vấn đề này, tách:

\[
 h_u^m \mapsto (z_{u}^{m,\mathrm{shared}},\,z_{u}^{m,\mathrm{private}}).
\]

- **Shared:** emotion-semantic evidence có thể căn chỉnh giữa modalities.
- **Private:** acoustic/visual/text-specific cues và sự bất đồng giữa modalities.

Chỉ áp dụng semantic alignment/OT lên phần shared; cho classification head tiếp cận **cả hai** phần. Đây là mở rộng đề xuất, **không** phải xác nhận rằng CAMP gốc sử dụng đúng decomposition này.

### 3.6. [PROPOSAL] Class-wise evidence fusion

Ví dụ evidence score của một modality đối với emotion \(e\):

\[
 \ell_{u,e}^m=\tau\log\left(\frac1J\sum_{j=1}^J
 \exp\big(\operatorname{cos}(z_u^m,p_{e,j}^m)/\tau\big)\right).
\]

Gating theo từng utterance và emotion:

\[
\ell_{u,e}=\sum_m g_{u,e}^{m}\ell_{u,e}^{m},\qquad
 g_{u,e}^m\ge0,\quad\sum_mg_{u,e}^m=1.
\]

Nhờ đó, audio/visual có thể đóng góp nhiều hơn text trong một utterance khi các cue không đồng nhất. Cần kiểm tra tính ổn định của gate, nhất là đối với minority classes.

---

## 4. Emotion wheel: Gắn vào đâu để phù hợp với cả hai bài?

**Research hypothesis:** Hình học psychological và hình học tối ưu cho emotion discrimination có thể xung đột. Nên áp đặt wheel ở cấp **semantic anchors** thay vì trực tiếp lên mọi instance feature hoặc attention query.

Tạo shared anchors \(Q=\{q_1,\ldots,q_K\}\), mỗi anchor thuộc một emotion class. Đặt \(S_{e,e'}^{\mathrm{wheel}}\) là similarity prior xây từ valence–arousal coordinates/relationships có căn cứ.

**[PROPOSAL] Soft wheel regularization:**

\[
\mathcal L_{\mathrm{wheel}}
=\frac{1}{|\mathcal E|}\sum_{(e,e')\in\mathcal E}
 w_{ee'}\big(\operatorname{cos}(q_e,q_{e'})-S_{ee'}^{\mathrm{wheel}}\big)^2.
\]

Ở đây \(\mathcal E\) là tập các emotion pairs có prior đáng tin, \(w_{ee'}\) cho phép giảm ảnh hưởng của các pair có quan hệ không chắc chắn.

**Không nên tùy tiện gán một góc cố định trên wheel cho Neutral.** Với Surprise, valence có thể thay đổi theo tình huống; có thể (a) bỏ các pair không chắc chắn khỏi wheel loss, (b) xây uncertainty-weighted relations, hoặc (c) thử học intensity/valence–arousal như auxiliary targets nếu có supervision hợp lệ.

Có thể kết hợp anchor-based classification:

\[
\mathcal L_{\mathrm{anchor}}=-\log\frac{\exp(\operatorname{cos}(z_u,q_{y_u})/\tau)}
{\sum_e\exp(\operatorname{cos}(z_u,q_e)/\tau)}.
\]

**Hai nhóm vector không được đánh đồng:** \(p_e^m\) (query/prototype dùng tìm evidence) và \(q_e\) (class-level semantic anchor được wheel-regularize).

---

## 5. Phương án tích hợp: 3 tầng rõ vai trò

```text
Utterance u + dialogue context
    |
    +--- Text encoder   ---> H_u^text  ---+
    +--- Audio encoder  ---> H_u^audio ----+----> [Layer 1: UniMER-inspired]
    +--- Visual encoder ---> H_u^visual ---+       Emotion-specific queries
                                                   c_{u,e}^m = Attn(p_e^m, H_u^m)
                                                              |
                                             [Layer 2: CAMP-inspired]
                                             Modality-specific evidence / prototypes
                                             Shared-private decomposition
                                             Optional class-wise OT on shared features
                                             Class-/utterance-dependent fusion
                                                              |
                                             [Layer 3: Emotion-wheel anchors]
                                             q_e + soft relational constraints
                                                              |
                                             Final class logits + prediction
```

### 5.1. Sự phân công của từng component

| Component | Đầu vào | Đầu ra | Vai trò |
|---|---|---|---|
| Emotion-specific attention | \(H_u^m,p_e^m\) | \(c_{u,e}^m\) | Tìm bằng chứng liên quan tới class \(e\) |
| Modality-specific prototype learning | \(c_{u,e}^m\) | \(P_e^m\), scores | Biểu diễn variations theo modality |
| Shared/private + OT | \(P_e^m\) | Aligned shared evidence | Căn chỉnh không cứng nhắc |
| Emotion semantic anchors | \(q_e,S^{wheel}\) | Geometry regularizer | Giữ inter-class psychological relations |
| Prediction head | Context + multimodal evidence | \(\hat y_u\) | Phân loại emotion của utterance |

### 5.2. Training loss đề xuất (không thuộc CAMP hoặc UniMER nguyên bản)

\[
\boxed{
\mathcal L=\mathcal L_{\mathrm{MERC}}
+\lambda_c\mathcal L_{\mathrm{class\ evidence}}
+\lambda_a\mathcal L_{\mathrm{anchor}}
+\lambda_o\mathcal L_{\mathrm{OT}}
+\lambda_w\mathcal L_{\mathrm{wheel}}
+\lambda_d\mathcal L_{\mathrm{diversity}}
}
\]

- \(\mathcal L_{\mathrm{MERC}}\): class-weighted CE / focal-like alternative (tune và báo cáo rõ).
- \(\mathcal L_{\mathrm{class\ evidence}}\): auxiliary supervision cho class-specific evidence channels; ngăn query specialization chỉ mang tính tên gọi.
- \(\mathcal L_{\mathrm{anchor}}\): gắn utterance representation vào class semantic anchor.
- \(\mathcal L_{\mathrm{OT}}\): class-wise alignment trên shared components; chỉ thêm khi có multi-prototype structure.
- \(\mathcal L_{\mathrm{wheel}}\): soft psychological relationships giữa anchors.
- \(\mathcal L_{\mathrm{diversity}}\): giảm redundant prototypes nếu có collapse.

**Chiến lược:** Bật từng loss theo ablation, không bật tất cả ngay từ đầu. Standardize feature scales, kiểm tra gradient magnitudes; tránh để \(\mathcal L_{\mathrm{wheel}}\) hoặc OT overwhelm classification loss.

---

## 6. Research questions và falsifiable hypotheses

**RQ1 — Class specificity:** Emotion-conditioned attention có học được các cues khác nhau cho các emotion classes, hay chỉ tạo một set queries dư thừa?

- **H1:** UniMER-style class attention cải thiện Macro-F1 và minority-class recall so với global pooling.
- **Potential failure:** Queries không có semantics ổn định; head chung vẫn làm tốt nhưng intervention không class-specific.

**RQ2 — Complementary evidence:** Raw feature alignment có làm giảm khả năng học multimodal cues không đồng nhất?

- **H2:** Shared-only OT/alignment + private evidence tốt hơn hard alignment trên toàn bộ feature với subset có cross-modal disagreement.
- **Potential failure:** Các modalities trên dataset quá redundant; OT làm tăng phức tạp mà không có lợi.

**RQ3 — Psychology vs. discrimination:** Soft emotion-wheel geometry ở anchor level có tốt hơn ép trực tiếp lên instance features không?

- **H3:** Anchor-level soft wheel constraint bảo toàn emotion relation structure mà không làm giảm fine-grained Macro-F1 như hard instance-level constraint.
- **Potential failure:** Wheel prior không phù hợp với nhãn dataset, hoặc chỉ tạo geometry đẹp nhưng không cải thiện classification.

**RQ4 — Joint benefit:** Việc kết hợp class-specific cues với complementary alignment có lợi hơn từng thành phần riêng lẻ không?

- **H4:** E1+E4 tăng hiệu quả trên utterances mà modalities bất đồng, đặc biệt ở minority emotions.

---

## 7. Ablation plan: ưu tiên nhanh → phức tạp

| ID | Thiết lập | Biến cần giữ cố định | Điều cần đo |
|---|---|---|---|
| **E0** | Existing MERC backbone + global pooling | Backbone, split, labels | Baseline Macro-F1 / Weighted-F1 |
| **E1** | E0 + UniMER-style class queries | Same encoders | Query contribution; minority-F1 |
| **E2** | E1 + anchor classification, **no wheel** | Same \(q_e\) capacity | Lợi ích do anchors hay do wheel? |
| **E3** | E2 + soft wheel regularization | Same param count | Geometry–discrimination trade-off |
| **E4** | E1 + modality-specific queries / evidence | No wheel, no OT | Có lợi từ native modality features? |
| **E5** | E4 + shared/private decomposition | Same feature dimension budget | Complementarity |
| **E6** | E5 + simple cosine/class alignment | Shared features | So sánh với OT |
| **E7** | E5 + class-wise OT với \(J>1\) | Same number of prototypes | OT vs direct matching |
| **E8** | E7 + anchors + soft wheel | Same training schedule | Joint benefit |

**Ablation bổ sung cần thiết:**

1. \(J=1,2,4\) prototypes per class/modality; kiểm tra utilization và collapse.
2. No alignment vs full-feature alignment vs shared-only alignment vs shared-only OT.
3. No wheel vs instance-level hard wheel vs instance-level soft wheel vs anchor-level soft wheel.
4. Shared queries vs separate modality queries; global pooling vs emotion-specific pooling.
5. Auxiliary class-evidence supervision on/off; performance khi bỏ từng modality.

**Đừng đánh đồng những kết quả từ khác backbone hoặc khác dataset.** Nếu làm comparison với UniMER đầy đủ, cần nói rõ khác biệt caption-based vs native modality encoders.

---

## 8. Evaluation beyond weighted F1

### Classification

- **Macro-F1**, Weighted-F1, per-class precision/recall/F1.
- Confusion matrix, đặc biệt giữa Fear, Anger, Sadness, Disgust, Surprise.
- Bootstrap confidence intervals hoặc multi-seed mean/std.

### Prototype / semantic geometry

- Pairwise anchor similarity matrix trước và sau wheel regularization.
- Spearman correlation giữa learned prototype similarities và \(S^{wheel}\) **trên các pair có prior đáng tin**.
- Inter-class margins, intra-class compactness, neighborhood preservation.
- Prototype usage histogram, pairwise cosine redundancy, dead prototype rate.

### Class-attention faithfulness

- Entropy / overlap của per-class attention distributions.
- Class-targeted prototype ablation: mask \(p_e\), đo biến động logit của \(e\) so với các classes khác.
- Qualitative examples: token/frame regions mà các queries tập trung; tránh kết luận attention là explanation nhân quả nếu chưa có intervention.

### Multimodal complementarity

- Text-only, audio-only, visual-only, pairwise modalities, all modalities.
- Modality-drop test và corruption/noise robustness.
- Subset có apparent cross-modal disagreement (cần định nghĩa trước, tránh lọc dựa trên output của model cuối cùng).
- Gate entropy và dominance: model có lạm dụng text hoặc một modality không?

---

## 9. Minimum viable experiment (khuyến nghị bắt đầu)

Để xác định liệu hướng này đáng đầu tư mà **không** cần implement toàn bộ CAMP/OT:

1. **Freeze hoặc fine-tune nhẹ backbone** đã có và lưu multimodal hidden states cho từng utterance.
2. **E0 vs E1:** Thay global pooling bằng emotion-specific attention queries; giữ classifier comparable.
3. **E1 vs E2 vs E3:** Thêm semantic anchors, sau đó thêm soft wheel loss **chỉ trên anchors**.
4. **E1 vs E4:** Kiểm tra emotion-specific queries riêng theo modality; đo minority-F1 và cross-modal disagreement subset.
5. Nếu E4 có lợi và có bằng chứng về heterogeneous modes, mở rộng \(J=2,4\) và thử shared-only alignment / class-wise OT.

**Go/no-go criteria:**

- E1 không giúp minority classes và không có query specialization → chưa có cơ sở tăng complexity bằng UniMER-style mechanism.
- E3 chỉ tăng wheel correlation nhưng giảm Macro-F1 rõ rệt → điều chỉnh prior/weighting; không tuyên bố thành công về emotion recognition.
- E4/E5 không cải thiện trên disagreement subset → OT chưa có động lực thực nghiệm.
- E7 không hơn simple alignment khi đã kiểm soát params → OT có thể không cần thiết.

---

## 10. Research novelty: điều gì đủ mạnh và điều gì chưa đủ?

**Chưa đủ mạnh nếu chỉ:** lấy class prototypes từ UniMER, thêm class-wise OT từ CAMP, cộng wheel MSE/contrastive loss rồi báo cáo F1 tăng nhẹ.

**Có thể mạnh hơn nếu chứng minh được:**

1. Có **representation failure** cụ thể (hard multimodal alignment xóa modality-specific emotion evidence; hoặc wheel-geometry constraints phá discrimination).
2. Có giải pháp **tách chức năng hợp lý**: query để lấy cues, modality prototype distributions để preserve complementary evidence, semantic anchors để giữ psychology.
3. Có ablation và diagnostics cho thấy **vì sao** giải pháp hiệu quả, không chỉ performance tổng.

**Research question đề xuất:**

> Can emotion prototypes preserve psychologically meaningful inter-class relationships while selectively aligning complementary multimodal evidence, without sacrificing fine-grained emotion discrimination?

**Research hypothesis đề xuất:**

> Emotion-wheel constraints should act on a shared class-level semantic anchor space, while emotion-conditioned modality features retain flexible discriminative and complementary evidence.

---

## 11. References

1. **UniMER** (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.1611/
2. **CAMP** (ICML 2026): https://proceedings.mlr.press/v306/pellicer26a.html
3. **Mapping the Circumplex of Affect** (ACL 2026): https://aclanthology.org/2026.acl-long.772/
4. **Prototype-as-Prompt** (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Prototype-as-Prompt_Multimodal_Sentiment_Prototypes_Endowing_Large_Language_Models_the_Capability_CVPR_2026_paper.html
5. **Neural Collapse by Design** (ICML 2026): https://proceedings.mlr.press/v306/koromilas26b.html
6. **Why Prototypes Collapse** (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/da42cd1440728b05f60f96a4a029965e-Abstract-Conference.html

**Ghi chú phương pháp:** Trước khi triển khai bản reproduction có tham vọng so sánh chính xác với CAMP, cần đối chiếu từng equation, component và training detail trực tiếp với PDF gốc và source code (nếu có). Những công thức được đánh dấu **[PROPOSAL]** là một thiết kế tổng hợp để thử nghiệm cho MERC, không phải cam kết tái hiện nguyên bản hai paper.
