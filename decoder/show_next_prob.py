import torch
import torch.nn.functional as F
from MiniGPT import MiniGPT 
from typing import Dict


def show_next_token_probs(
        model_before : MiniGPT, 
        model_after : MiniGPT,
        text : str, 
        stoi : Dict[str, int], 
        itos : Dict[int, str]
    ):
    
    """
    text を与えて、各ステップで:
    - 正解トークンの確率
    を「学習前 → 学習後」で比較して表示する
    """

    # 文字列を ID に変換
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long).unsqueeze(0)

    # 各タイムステップごとに処理
    for t in range(len(text) - 1):
        prefix = ids[:, :t+1]          # 入力（文脈）
        target_id = ids[0, t+1].item() # 正解トークン ID
        
        # --- 学習前モデル ---
        with torch.no_grad():
            logits_before = model_before(prefix)[:, -1, :] 
            probs_before = F.softmax(logits_before, dim=-1).squeeze()

        # --- 学習後モデル ---
        with torch.no_grad():
            logits_after = model_after(prefix)[:, -1, :]
            probs_after = F.softmax(logits_after, dim=-1).squeeze()

        # 正解トークンの文字
        target_char = itos[target_id]

        # 可視化
        print(f"===== step {t} =====")
        print(f"Context: {text[:t+1]}")
        print(f"next token: '{target_char}'")

        print(f"P('{target_char}' | context) BEFORE = {probs_before[target_id]:.4f}")
        print(f"P('{target_char}' | context) AFTER  = {probs_after[target_id]:.4f}")
