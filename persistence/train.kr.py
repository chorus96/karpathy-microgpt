"""
문자 단위 데이터셋으로 GPT를 학습하고, 학습된 가중치를 디스크에 저장한다.

Karpathy의 microgpt를 기반으로 한다 -- 동일한 243줄 알고리즘에, 재학습 없이
모델을 재사용할 수 있도록 영속성(persistence)을 더했다.

사용법: uv run train.py [--steps 500] [--output model.json]

[이 파일은 persistence/train.py의 한국어 주석 버전입니다. 코드는 원본과
동일하며, 주석만 한국어로 옮기고 이해를 돕는 설명을 덧붙였습니다.]
"""

import os
import sys
import math
import json
import random

# ---------------------------------------------------------------------------
# CLI 인자 (의존성 없이 유지 -- 플래그가 둘뿐이라 argparse가 필요 없다)
# ---------------------------------------------------------------------------
num_steps = 500          # 기본 학습 스텝 수
output_path = 'model.json'  # 기본 저장 경로

args = sys.argv[1:]  # 프로그램 이름을 뺀 실제 인자들
i = 0
while i < len(args):
    if args[i] == '--steps' and i + 1 < len(args):
        num_steps = int(args[i + 1]); i += 2       # --steps 다음 값을 정수로
    elif args[i] == '--output' and i + 1 < len(args):
        output_path = args[i + 1]; i += 2          # --output 다음 값을 경로로
    else:
        print(f"Unknown arg: {args[i]}"); sys.exit(1)  # 알 수 없는 인자는 오류 종료

# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
random.seed(42)  # 재현성을 위한 시드 고정

if not os.path.exists('input.txt'):
    # input.txt가 없으면 makemore 이름 데이터를 내려받는다
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)  # 순서를 섞어 학습 편향 방지
print(f"num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# 토크나이저 (문자 단위, BOS 구분자 포함)
# ---------------------------------------------------------------------------
chars = ['<BOS>'] + sorted(set(''.join(docs)))  # 고유 문자 정렬 + BOS
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) }   # 문자 → 정수(인코딩)
itos = { i:ch for i, ch in enumerate(chars) }   # 정수 → 문자(디코딩)
BOS = stoi['<BOS>']
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# 자동 미분 엔진 (Karpathy 원본과 동일)
# ---------------------------------------------------------------------------
class Value:
    """하나의 스칼라 값과 그 기울기를 저장한다."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data                # 실제 값
        self.grad = 0                   # 손실에 대한 이 값의 기울기
        self._backward = lambda: None   # 이 노드의 역전파 규칙(기본: 무동작)
        self._prev = set(_children)     # 이 값을 만든 자식 노드들
        self._op = _op                  # 이 노드를 만든 연산(디버깅용)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # 덧셈의 미분: 기울기를 양쪽에 그대로 전달
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 곱의 법칙: 상대편 값을 곱해 전달
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            # 거듭제곱 법칙: d(a^n)/da = n*a^(n-1)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            # 로그의 미분: 1/a
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            # 지수의 미분: e^a (자기 자신)
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            # ReLU의 미분: 양수면 1, 아니면 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []               # 위상 정렬 결과
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1           # 출력(손실)의 기울기는 1
        for v in reversed(topo):  # 역순으로 연쇄 법칙 적용
            v._backward()

    # 편의를 위한 연산자 정의들
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

# ---------------------------------------------------------------------------
# 모델 파라미터
# ---------------------------------------------------------------------------
n_embd = 16      # 임베딩 차원
n_head = 4       # 어텐션 헤드 수
n_layer = 1      # 레이어 수
block_size = 8   # 컨텍스트 길이
head_dim = n_embd // n_head  # 헤드당 차원(4)
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)           # Query 투영
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)           # Key 투영
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)           # Value 투영
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)    # 출력 투영
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)       # MLP 확장
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)  # MLP 축소
params = [p for mat in state_dict.values() for row in mat for p in row]  # 전체 파라미터 평탄화
print(f"num params: {len(params)}")

# ---------------------------------------------------------------------------
# 모델 아키텍처 (gpt.py와 동일)
# ---------------------------------------------------------------------------
def linear(x, w):
    # 선형 변환(행렬-벡터 곱)
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 로짓 → 확률. 최댓값 빼기로 수치 안정화.
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # RMSNorm: 제곱평균제곱근으로 정규화
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # 한 토큰 위치에 대한 순전파
    tok_emb = state_dict['wte'][token_id]  # 토큰 임베딩
    pos_emb = state_dict['wpe'][pos_id]    # 위치 임베딩
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 둘을 더함
    x = rmsnorm(x)

    for li in range(n_layer):
        # 멀티헤드 어텐션 블록
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)      # KV 캐시에 K 추가
        values[li].append(v)    # KV 캐시에 V 추가
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # Q·K 내적 / sqrt(head_dim) = 어텐션 점수
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
        x = [xi.relu() ** 2 for xi in x]            # ReLU^2 활성화
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # 잔차 연결

    logits = linear(x, state_dict['lm_head'])
    return logits

# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # Adam 1차 모멘트 버퍼
v = [0.0] * len(params)  # Adam 2차 모멘트 버퍼

print(f"\nTraining for {num_steps} steps...")
for step in range(num_steps):
    doc = docs[step % len(docs)]  # 순환 인덱스로 문서 선택
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]  # 양 끝에 BOS
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # KV 캐시 초기화
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 교차 엔트로피
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # 평균 손실

    loss.backward()  # 역전파

    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감소
    for i, p in enumerate(params):
        # Adam 갱신
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 기울기 초기화

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ---------------------------------------------------------------------------
# 학습된 모델 저장 (이 스크립트의 고유 부분)
# ---------------------------------------------------------------------------
model = {
    # 하이퍼파라미터 -- 로드 시 아키텍처를 재구성하는 데 필요
    'config': {
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'vocab_size': vocab_size,
    },
    # 토크나이저 매핑 -- 텍스트 인코딩/디코딩에 필요
    'chars': chars,
    # 학습된 가중치 -- 모델의 실제 지식 (Value 래퍼를 벗기고 순수 float만 저장)
    'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
}

with open(output_path, 'w') as f:
    json.dump(model, f)  # JSON으로 직렬화

size_kb = os.path.getsize(output_path) / 1024
print(f"\nModel saved to {output_path} ({size_kb:.1f} KB)")
print(f"Contains: config, tokenizer ({vocab_size} tokens), weights ({len(params)} params)")
