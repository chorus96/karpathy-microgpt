"""
디스크에 저장된 GPT를 불러와 텍스트를 생성한다. 학습도, 기울기도 없이 --
저장된 가중치로 순전파 추론만 수행한다.

사용법: uv run run.py [--model model.json] [--samples 20] [--temperature 0.6]

[이 파일은 persistence/run.py의 한국어 주석 버전입니다. 코드는 원본과
동일하며, 주석만 한국어로 옮기고 이해를 돕는 설명을 덧붙였습니다.]
"""

import sys
import math
import json
import random

# ---------------------------------------------------------------------------
# CLI 인자
# ---------------------------------------------------------------------------
model_path = 'model.json'  # 불러올 모델 경로
num_samples = 20           # 생성할 이름 개수
temperature = 0.6          # 창의성 조절(낮을수록 보수적)

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--model' and i + 1 < len(args):
        model_path = args[i + 1]; i += 2
    elif args[i] == '--samples' and i + 1 < len(args):
        num_samples = int(args[i + 1]); i += 2       # 문자열 → 정수
    elif args[i] == '--temperature' and i + 1 < len(args):
        temperature = float(args[i + 1]); i += 2     # 문자열 → 실수
    else:
        print(f"Unknown arg: {args[i]}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 디스크에서 모델 로드
# ---------------------------------------------------------------------------
print(f"Loading model from {model_path}...")

with open(model_path, 'r') as f:
    model = json.load(f)  # JSON을 파이썬 딕셔너리로

config = model['config']    # 하이퍼파라미터
chars = model['chars']      # 토크나이저 문자 목록
weights = model['weights']  # 학습된 가중치(순수 float)

# config에서 하이퍼파라미터를 그대로 읽어 동일한 아키텍처를 복원
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
block_size = config['block_size']
vocab_size = config['vocab_size']
head_dim = n_embd // n_head

# 토크나이저 재구성 (가중치와 반드시 함께 이동해야 함)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
BOS = stoi['<BOS>']

# state_dict를 순수 float 리스트로 복원 (추론에는 기울기가 없으므로 Value 래퍼 불필요)
state_dict = {k: [list(row) for row in mat] for k, mat in weights.items()}

num_params = sum(len(row) for mat in state_dict.values() for row in mat)
print(f"Loaded: {vocab_size} tokens, {num_params} params, context length {block_size}")

# ---------------------------------------------------------------------------
# 모델 아키텍처 (추론 전용, 순수 float -- autograd 오버헤드 없음)
# ---------------------------------------------------------------------------
def linear(x, w):
    # 선형 변환(행렬-벡터 곱)
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 로짓 → 확률. 학습판과 달리 Value가 아닌 순수 float를 다룬다.
    max_val = max(logits)                            # float 직접 비교
    exps = [math.exp(v - max_val) for v in logits]   # math.exp 사용
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # RMSNorm: 제곱평균제곱근으로 정규화
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # 순전파 (구조는 학습판과 동일, 순수 float로 동작)
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 멀티헤드 어텐션 블록
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)      # KV 캐시
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결
        # MLP 블록
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [max(0, xi) ** 2 for xi in x]  # ReLU^2 (순수 float에는 내장 max 사용)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결

    logits = linear(x, state_dict['lm_head'])
    return logits

# ---------------------------------------------------------------------------
# 샘플 생성
# ---------------------------------------------------------------------------
print(f"\n--- inference (temperature={temperature}) ---")
for sample_idx in range(num_samples):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # 샘플마다 KV 캐시 초기화
    token_id = BOS  # BOS에서 시작
    result = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])  # 온도로 분포 조절
        token_id = random.choices(range(vocab_size), weights=probs)[0]  # 샘플링
        if token_id == BOS:
            break  # 다시 BOS면 이름 끝
        result.append(itos[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(result)}")
