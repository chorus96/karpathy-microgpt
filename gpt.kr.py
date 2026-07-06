"""
순수하고 의존성 없는 Python만으로 GPT를 학습하고 추론하는 가장 원자적인(atomic) 방법.
이 파일이 완전한 알고리즘이다. 나머지는 모두 효율성을 위한 것일 뿐이다.

@karpathy

[이 파일은 gpt.py의 한국어 주석 버전입니다. 코드는 원본과 동일하며, 주석만
한국어로 번역하고 이해를 돕는 설명을 덧붙였습니다.]
"""

import os       # os.path.exists (파일 존재 확인)
import math     # math.log, math.exp (로그·지수 함수)
import random   # random.seed, random.choices, random.gauss, random.shuffle (난수)

# 혼돈 속에 질서가 있으라 (창세기 문체의 위트).
# 시드를 고정해 매 실행이 동일하게 재현되도록 한다.
random.seed(42)

# 입력 데이터셋 `docs`가 있으라: 문서들의 list[str] (예: 이름 데이터셋)
# input.txt가 없으면 makemore의 이름 약 32,000개를 내려받는다.
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # 문서들의 list[str]
random.shuffle(docs)  # 학습 편향을 막기 위해 순서를 섞는다
print(f"num docs: {len(docs)}")

# 문자열을 이산 기호(정수)로, 다시 그 반대로 변환하는 토크나이저가 있으라
# BOS(<Beginning Of Sequence>) 구분자를 포함한 문자 단위 토크나이저.
chars = ['<BOS>'] + sorted(set(''.join(docs))) # 모든 문서에서 고유 문자만 추려 정렬 + BOS
vocab_size = len(chars)  # 어휘 크기 = 27 (문자 26 + BOS)
stoi = { ch:i for i, ch in enumerate(chars) } # 인코딩: 문자 → 정수
itos = { i:ch for i, ch in enumerate(chars) } # 디코딩: 정수 → 문자
BOS = stoi['<BOS>']  # BOS 토큰의 정수 ID
print(f"vocab size: {vocab_size}")

# 계산 그래프 전체에 연쇄 법칙을 재귀적으로 적용해, 모델 파라미터에 대한 손실의
# 기울기(gradient)를 계산하는 자동 미분(Autograd) 엔진이 있으라.
class Value:
    """하나의 스칼라 값과 그 기울기를 저장한다."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data                # 실제 값
        self.grad = 0                   # 손실에 대한 이 값의 기울기 (역전파로 채워짐)
        self._backward = lambda: None   # 이 노드의 역전파 규칙 (기본: 아무것도 안 함)
        self._prev = set(_children)     # 이 값을 만든 입력(자식) 노드들
        self._op = _op # 이 노드를 만든 연산 (graphviz / 디버깅용)

    def __add__(self, other):
        # 덧셈: out = self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # 덧셈의 미분: 기울기를 양쪽에 그대로 전달 (d(a+b)/da = 1)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        # 곱셈: out = self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 곱의 법칙: 상대편 값을 곱해 전달 (d(a*b)/da = b)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        # 거듭제곱: out = self ** other (지수는 상수만 허용)
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            # 거듭제곱 법칙: d(a^n)/da = n * a^(n-1)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        # 자연로그: out = ln(self)
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            # 로그의 미분: d(ln a)/da = 1/a
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        # 지수함수: out = e^self
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            # 지수의 미분: d(e^a)/da = e^a (= out.data 자기 자신)
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # ReLU: 음수는 0으로, 양수는 그대로
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            # ReLU의 미분: 입력이 양수면 1, 아니면 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # 그래프의 모든 자식 노드를 위상 정렬(topological order)한다
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 한 변수씩 연쇄 법칙을 적용해 기울기를 구한다
        self.grad = 1  # 출력(손실) 자신에 대한 기울기는 1
        for v in reversed(topo):  # 역위상 순서(출력 → 입력)로 전파
            v._backward()

    # 아래는 편의를 위한 연산자 정의들 (뺄셈·나눗셈 등을 위/오른쪽 피연산자까지 지원)
    def __neg__(self): return self * -1                       # -self
    def __radd__(self, other): return self + other            # other + self
    def __sub__(self, other): return self + (-other)          # self - other
    def __rsub__(self, other): return other + (-self)         # other - self
    def __rmul__(self, other): return self * other            # other * self
    def __truediv__(self, other): return self * other**-1     # self / other
    def __rtruediv__(self, other): return other * self**-1    # other / self
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

# 모델의 지식을 담을 파라미터를 초기화한다.
n_embd = 16     # 임베딩 차원
n_head = 4      # 어텐션 헤드 수
n_layer = 1     # 레이어(트랜스포머 블록) 수
block_size = 8  # 최대 시퀀스 길이(컨텍스트 길이)
head_dim = n_embd // n_head # 헤드 하나의 차원 (16 / 4 = 4)
# nout x nin 크기의 가중치 행렬을 만드는 팩토리 (작은 가우시안 난수로 초기화)
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# wte: 토큰 임베딩, wpe: 위치 임베딩, lm_head: 출력 헤드
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)          # 어텐션 Query 투영
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)          # 어텐션 Key 투영
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)          # 어텐션 Value 투영
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)   # 어텐션 출력 투영 (0에서 시작 → 잔차 안정화)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)      # MLP 확장 (16→64)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0) # MLP 축소 (64→16, 0에서 시작)
params = [p for mat in state_dict.values() for row in mat for p in row] # 전체 파라미터를 한 리스트로 평탄화
print(f"num params: {len(params)}")

# 모델 아키텍처 정의: 토큰 시퀀스와 파라미터를 받아 "다음에 올 것"에 대한 로짓을 내는 무상태 함수.
# GPT-2를 따르되 약간의 차이: layernorm → rmsnorm, 편향 없음, GeLU → ReLU^2
def linear(x, w):
    # 선형 변환(행렬-벡터 곱): 각 출력 뉴런마다 가중치와 입력의 내적
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 로짓 → 확률 분포. 최댓값을 빼서 지수 오버플로를 막는다(수치 안정화).
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # RMSNorm: 벡터를 제곱평균제곱근으로 나눠 크기를 정규화 (LayerNorm보다 단순)
    ms = sum(xi * xi for xi in x) / len(x)   # 제곱들의 평균(mean square)
    scale = (ms + 1e-5) ** -0.5              # 1/sqrt(ms) (1e-5는 0 나눗셈 방지)
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # 한 토큰 위치에 대한 순전파. keys/values는 이전 위치들의 KV 캐시.
    tok_emb = state_dict['wte'][token_id] # 토큰 임베딩
    pos_emb = state_dict['wpe'][pos_id] # 위치 임베딩
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 토큰+위치 임베딩을 더함
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) 멀티헤드 어텐션 블록
        x_residual = x           # 잔차 연결을 위해 입력 보관
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value
        keys[li].append(k)       # KV 캐시에 현재 위치의 K 추가
        values[li].append(v)     # KV 캐시에 현재 위치의 V 추가
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim                          # 이 헤드가 차지하는 구간의 시작
            q_h = q[hs:hs+head_dim]                     # 이 헤드의 Query 조각
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 지금까지의 모든 K 조각
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 지금까지의 모든 V 조각
            # 어텐션 점수: Q·K 내적을 sqrt(head_dim)으로 나눔(스케일링)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)         # 점수 → 가중치
            # 가중치로 V들을 가중 합
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)                     # 헤드 결과를 이어 붙임
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 출력 투영
        x = [a + b for a, b in zip(x, x_residual)]      # 잔차 연결(+ 입력)
        # 2) MLP 블록
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 확장 (16→64)
        x = [xi.relu() ** 2 for xi in x]                 # ReLU^2 활성화(비선형)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 축소 (64→16)
        x = [a + b for a, b in zip(x, x_residual)]       # 잔차 연결

    logits = linear(x, state_dict['lm_head'])  # 어휘 크기(27)의 로짓으로 투영
    return logits

# 축복받은 옵티마이저 Adam과 그 버퍼가 있으라
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params) # 1차 모멘트(기울기 이동평균) 버퍼
v = [0.0] * len(params) # 2차 모멘트(기울기 제곱 이동평균) 버퍼

# 순서대로 반복하라
num_steps = 500 # 학습 스텝 수
for step in range(num_steps):

    # 문서(이름) 하나를 골라 토큰화하고, 양 끝에 BOS 특수 토큰을 붙인다
    doc = docs[step % len(docs)]   # 순환 인덱스로 문서 선택
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # 컨텍스트 길이로 제한

    # 토큰 시퀀스를 모델에 순전파하며, 손실까지 이르는 계산 그래프를 구축한다.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # KV 캐시 초기화
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 입력 토큰과 정답(다음 토큰)
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # 교차 엔트로피: -log(정답 확률)
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # 문서 시퀀스 전체의 평균 손실. 낮기를 바란다.

    # 손실을 역전파해, 모든 파라미터에 대한 기울기를 계산한다.
    loss.backward()

    # Adam 옵티마이저 업데이트: 각 파라미터를 그 기울기에 따라 갱신한다.
    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감소
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad          # 1차 모멘트 갱신
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2     # 2차 모멘트 갱신
        m_hat = m[i] / (1 - beta1 ** (step + 1))            # 편향 보정
        v_hat = v[i] / (1 - beta2 ** (step + 1))            # 편향 보정
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)  # 파라미터 갱신
        p.grad = 0                                          # 다음 스텝을 위해 기울기 초기화

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# 추론: 모델이 우리에게 되뇌어 주기를
temperature = 0.6 # (0, 1] 범위. 생성 텍스트의 "창의성"을 낮음→높음으로 조절
print("\n--- inference ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # 샘플마다 KV 캐시 초기화
    token_id = BOS  # BOS에서 시작
    print(f"sample {sample_idx+1}: ", end="")
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])  # 온도로 분포의 뾰족함 조절
        # 확률 분포에서 다음 토큰을 샘플링
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break  # 다시 BOS가 나오면 이름 끝
        print(itos[token_id], end="")
    print()
