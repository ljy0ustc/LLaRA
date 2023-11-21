from torch import nn

class MlpProjector(nn.Module):
    def __init__(self, rec_size=64, llm_size=4096):
        super().__init__()
        self.mlp_proj = nn.Sequential(
            nn.Linear(rec_size, llm_size),
            nn.GELU(),
            nn.Linear(llm_size, llm_size)
        )

    def forward(self, x):
        x = self.mlp_proj(x)
        return x