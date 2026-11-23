
import torch
import torch.nn as nn

class SelfAttnBlock(nn.Module):
    def __init__(self, d=32, dropout=0.3):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # x:(B,D) 单token演示; 多token时可扩展为(B,L,D)+ScaleDotProduct
        a = self.o(self.v(x))
        a = self.drop(a)
        return self.ln(x + a)

class CrossAttnBlock(nn.Module):
    def __init__(self, d=32, dropout=0.3):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
    def forward(self, q_in, kv_in):
        # Q来自一模态, K/V来自另一模态
        _ = self.q(q_in) * self.k(kv_in)  # 相互作用(单token简化)
        a = self.o(self.v(kv_in))
        a = self.drop(a)
        return self.ln(q_in + a)

class TransModel(nn.Module):
    def __init__(self, d=32, hidden=64, dropout=0.3):
        super().__init__()
        self.sa_rad = SelfAttnBlock(d, dropout)
        self.sa_dl = SelfAttnBlock(d, dropout)
        self.sa_patho = SelfAttnBlock(d, dropout)
        self.ca1 = CrossAttnBlock(d, dropout)  # rad<-dl
        self.ca2 = CrossAttnBlock(d, dropout)  # rad<-patho
        self.ffn = nn.Sequential(
            nn.Linear(96, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 96),
        )
        self.ln = nn.LayerNorm(96)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(),
            nn.Linear(32, 1),  # logit
        )
    def forward(self, x_rad, x_dl, x_patho):
        assert x_rad.size(1) == 32 and x_dl.size(1) == 32 and x_patho.size(1) == 32
        h_r = self.sa_rad(x_rad)
        h_d = self.sa_dl(x_dl)
        h_p = self.sa_patho(x_patho)
        h_r1 = self.ca1(h_r, h_d)
        h_r2 = self.ca2(h_r, h_p)
        # 融合: 用cross增强的rad + dl + patho -> 96维
        F = torch.cat([h_r1, h_d, h_p], dim=1)  # (B,96), 若用h_r2可替换其中之一
        F2 = self.ln(F + self.drop(self.ffn(F)))
        logit = self.fc(F2)  # (B,1)
        return logit

def build_loss(y_train, device="cpu"):
    # class-balanced pos_weight = neg/pos
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

if __name__ == "__main__":
    # ===== 矩阵测试 (需torch环境) =====
    device = "cpu"
    m = TransModel().to(device)
    m.train()
    B = 8
    xr = torch.randn(B, 32); xd = torch.randn(B, 32); xp = torch.randn(B, 32)
    y = torch.randint(0, 2, (B, 1)).float()
    logit = m(xr, xd, xp)
    assert logit.shape == (B, 1), logit.shape
    loss_fn = build_loss(y.view(-1))
    loss = loss_fn(logit.view(-1), y.view(-1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-2)
    opt.zero_grad(); loss.backward(); opt.step()
    print("logit:", tuple(logit.shape), "loss:", float(loss))
    m.eval()
    with torch.no_grad():
        for b in [1, 16]:
            o = m(torch.randn(b, 32), torch.randn(b, 32), torch.randn(b, 32))
            assert o.shape == (b, 1)
            print(f"batch {b} OK -> {tuple(o.shape)}")
    print("ALL TORCH MATRIX TESTS PASSED")
