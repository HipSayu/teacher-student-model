# Chặng 3 — `CAKD/cakd_modified_files/vision_transformer.py` (GIÁO VIÊN)

> ViT-B/16 gốc của torchvision, **đóng băng** khi train. CAKD chỉ sửa 3 chỗ trong `forward` để
> **moi ra** attention map + token + feature làm "đáp án mẫu". Đọc kỹ 3 chỗ đó.

---

## Bối cảnh: giáo viên phải trả về 4 thứ (khớp với học sinh)
```python
return x, attn_weights, cls_token, feats   # dòng 318
#      logits   4 map      token CLS   đặc trưng patch
```

---

## 🟢 `EncoderBlock.forward` — dòng 110–121

Một block Transformer. CAKD thêm tham số `need_weights` để **tùy chọn trả về attention map**.

```python
def forward(self, input, need_weights: bool=False):        # ← THÊM cờ need_weights
    torch._assert(input.dim() == 3, ...)
    x = self.ln_1(input)                                    # LayerNorm trước attention
    x, attn_weights = self.self_attention(query=x, key=x, value=x, need_weights=need_weights)  # self-attention
    x = self.dropout(x)
    x = x + input                                           # residual 1
    y = self.ln_2(x)                                        # LayerNorm trước MLP
    y = self.mlp(y)                                         # MLP
    if not need_weights:
        return x + y                                        # block thường: chỉ trả đặc trưng
    return x + y, attn_weights                              # ← khi cần: trả THÊM attention map
```
> `self.self_attention` là `nn.MultiheadAttention`. Khi `need_weights=True` nó trả kèm ma trận trọng số chú ý.
> (Bản CAKD đã chỉnh để `attn_weights` mang được cả 2 map qk & vv — song song với `Attention` bên học sinh.)

---

## 🟢 `Encoder.forward` — dòng 156–169  (CHỖ QUAN TRỌNG NHẤT của teacher)

Chạy chồng nhiều block, nhưng **chỉ moi attention của 2 block CUỐI**.

```python
def forward(self, input):
    input = input + self.pos_embedding                     # cộng vị trí (positional embedding)
    num_layers = len(self.layers)
    x = self.dropout(input)
    for i in range(num_layers):
        if i < num_layers - 2:
            x = self.layers[i](x)                          # các block đầu: chạy thường, KHÔNG lấy weight
        elif i == num_layers - 2:
            x, attn_weights_2 = self.layers[i](x, True)    # block ÁP CHÓT: lấy attention (need_weights=True)
        else:
            x, attn_weights_1 = self.layers[i](x, True)    # block CUỐI:    lấy attention
    return self.ln(x), [attn_weights_2[0], attn_weights_2[1], attn_weights_1[0], attn_weights_1[1]]
    #                    └── áp chót: qk,vv ──┘  └──── cuối: qk,vv ────┘
    #                        index 0    1            index 2    3
```
> **Danh sách trả về có 4 phần tử** — nhớ index này vì train dùng trực tiếp:
> | index | là gì |
> |-------|-------|
> | `[0]` | qk của block **áp chót** |
> | `[1]` | vv của block **áp chót** |
> | `[2]` | qk của block **cuối** ← `pca_loss` khớp với `attn_qk` học sinh |
> | `[3]` | vv của block **cuối** ← `pca_loss` khớp với `attn_vv` học sinh |
>
> **Vì sao chỉ 2 block cuối?** Attention ở lớp sâu mang ngữ nghĩa "toàn cục" rõ nhất → đáng để học sinh bắt chước.
> Lấy hết mọi lớp thì thừa và tốn.

---

## 🟢 `VisionTransformer.forward` — dòng 301–318

```python
def forward(self, x):
    x = self._process_input(x)                             # ảnh → chuỗi patch embedding (B, 196, 768)
    n = x.shape[0]
    batch_class_token = self.class_token.expand(n, -1, -1) # token CLS nhân ra cả batch
    x = torch.cat([batch_class_token, x], dim=1)           # ghép CLS vào đầu → (B, 197, 768)
    x, attn_weights = self.encoder(x)                      # chạy encoder → đặc trưng + 4 attention map
    cls_token = x[:, 0]                                    # tách token CLS   → (B, 768)  ← "tea_token"
    feats = x[:, 1:]                                       # tách 196 patch  → (B,196,768) ← "tea_feat"
    x = self.heads(cls_token)                              # đầu phân loại từ CLS → (B, n_cls) ← "tea_logits"
    return x, attn_weights, cls_token, feats
    #      logits   4 map       token      patch-feat
```
> **Chú ý kích thước 197 = 1 CLS + 196 patch.** Đây là lý do trong train có đoạn `[:, 1:, 1:]` khi so với học sinh:
> phải **cắt bỏ hàng/cột CLS** để attention teacher còn 196×196, khớp đúng với 196 patch của học sinh (chính là chữ **"Partially"** trong PCA).

---

## ⚪ Phần LƯỚT
`ConvStemConfig`, `MLPBlock` (chỉ là tương thích state_dict cũ), `Encoder.__init__`, `VisionTransformer.__init__`,
`_process_input`, các `ViT_*_Weights`, factory `vit_b_16/...`. Đều là torchvision gốc.

Trong train, giáo viên nạp trọng số pretrained:
```python
teacher = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
teacher.eval()   # đóng băng — chỉ suy luận, mọi output .detach() trước khi vào loss
```

---

## ✅ Kiểm tra hiểu
1. `teacher(image)` trả về 4 thứ nào? Cái nào khớp với `proj_token`, cái nào khớp với `proj_feat` của học sinh?
2. Trong `tea_attn_weights`, tại sao train dùng index `[2]` và `[3]` chứ không phải `[0]`,`[1]`?
3. `[:, 1:, 1:]` dùng để làm gì và liên quan gì tới chữ "Partially"?
