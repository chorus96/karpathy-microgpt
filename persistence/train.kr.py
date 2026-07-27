"""
문자 단위 데이터셋으로 GPT를 학습하고, 학습된 가중치를 디스크에 저장한다.

Karpathy의 microgpt를 기반으로 한다 -- 동일한 243줄 알고리즘에, 재학습 없이
모델을 재사용할 수 있도록 영속성(persistence)을 더했다.

사용법: uv run train.py [--steps 500] [--output model.json]

[이 파일은 persistence/train.py의 한국어 주석 버전입니다. 코드는 원본과
동일하며, 각 줄을 자세히 설명하는 주석을 덧붙였습니다.]
"""

import os       # 파일 존재 확인(os.path.exists), 파일 크기(os.path.getsize)
import sys      # 명령행 인자(sys.argv), 프로그램 종료(sys.exit)
import math     # 로그(math.log)·지수(math.exp) 함수
import json     # 학습 결과를 model.json으로 직렬화
import random   # 난수: 시드 고정, 셔플, 가우시안 초기화

# ---------------------------------------------------------------------------
# CLI 인자 (의존성 없이 유지 -- 플래그가 둘뿐이라 argparse가 필요 없다)
# ---------------------------------------------------------------------------
num_steps = 500             # (기본값) 학습을 몇 스텝 돌지
output_path = 'model.json'  # (기본값) 학습된 모델을 저장할 파일 경로

args = sys.argv[1:]  # sys.argv[0]은 스크립트 이름이므로 [1:]로 실제 인자만 취함
i = 0                # 인자 리스트를 훑을 인덱스
while i < len(args):                                  # 인자를 앞에서부터 하나씩 검사
    if args[i] == '--steps' and i + 1 < len(args):   # '--steps'이고 뒤에 값이 있으면
        num_steps = int(args[i + 1]); i += 2         #   다음 값을 정수로 변환, 인덱스 2칸 전진
    elif args[i] == '--output' and i + 1 < len(args):# '--output'이고 뒤에 값이 있으면
        output_path = args[i + 1]; i += 2            #   다음 값을 경로 문자열로 저장
    else:                                            # 알 수 없는 인자를 만나면
        print(f"Unknown arg: {args[i]}"); sys.exit(1)#   오류 메시지 후 종료 코드 1로 중단

# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
random.seed(42)  # 난수 시드를 고정해 매 실행이 동일하게 재현되도록 함

if not os.path.exists('input.txt'):  # 학습 데이터 파일이 없으면
    # makemore의 이름 데이터(약 32k개)를 인터넷에서 내려받는다
    import urllib.request  # 표준 라이브러리 HTTP 다운로드 (여기서만 필요해 지역 임포트)
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')  # URL → 로컬 파일로 저장
# 파일을 읽어 줄 단위로 분리하고, 앞뒤 공백 제거 후 빈 줄은 걸러 문서 리스트를 만든다
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)              # 순서를 무작위로 섞어 학습이 특정 순서에 치우치지 않게 함
print(f"num docs: {len(docs)}")  # 불러온 문서(이름) 개수 출력

# ---------------------------------------------------------------------------
# 토크나이저 (문자 단위, BOS 구분자 포함)
# ---------------------------------------------------------------------------
# 모든 문서를 이어 붙여 등장하는 고유 문자만 추출(set)하고 정렬한 뒤, 맨 앞에 특수 토큰 <BOS>를 둔다
chars = ['<BOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)                          # 어휘 크기 = 문자 종류 수(여기선 27 = a~z + <BOS>)
stoi = { ch:i for i, ch in enumerate(chars) }    # string→int: 문자 → 정수 ID (인코딩 표)
itos = { i:ch for i, ch in enumerate(chars) }    # int→string: 정수 ID → 문자 (디코딩 표)
BOS = stoi['<BOS>']                              # <BOS> 토큰의 정수 ID를 미리 구해 둠
print(f"vocab size: {vocab_size}")              # 어휘 크기 출력

# ---------------------------------------------------------------------------
# 자동 미분 엔진 (Karpathy 원본과 동일)
# ---------------------------------------------------------------------------
class Value:
    """하나의 스칼라 값과 그 기울기를 저장한다."""
    # 각 Value 객체 = 계산 그래프의 노드 하나. 값(data)과, 손실에 대한 기울기(grad)를 가진다.

    def __init__(self, data, _children=(), _op=''):
        self.data = data                # 이 노드가 담는 실제 스칼라 값
        self.grad = 0                   # 손실 L에 대한 이 값의 편미분(∂L/∂data). 역전파로 채워짐
        self._backward = lambda: None   # 이 노드의 기울기를 자식들에게 전파하는 함수(기본은 무동작)
        self._prev = set(_children)     # 이 값을 만들어낸 입력(자식) 노드들의 집합
        self._op = _op                  # 이 노드를 만든 연산 이름(디버깅/시각화용)

    def __add__(self, other):
        # 덧셈 연산자(+) 정의: out = self + other
        other = other if isinstance(other, Value) else Value(other)  # 숫자면 Value로 감쌈
        out = Value(self.data + other.data, (self, other), '+')      # 결과 노드 생성
        def _backward():
            # 덧셈의 미분: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1 → 상류 기울기를 양쪽에 그대로 더함
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward  # 결과 노드에 역전파 규칙 부착
        return out

    def __mul__(self, other):
        # 곱셈 연산자(*) 정의: out = self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 곱의 법칙: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a → 상대편 값을 곱해 전파
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        # 거듭제곱 연산자(**) 정의: out = self ** other (지수는 상수만 지원)
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
            # 지수의 미분: d(e^a)/da = e^a → out.data가 곧 e^a이므로 그것을 곱함
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # ReLU: 음수는 0, 양수는 그대로 통과
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            # ReLU의 미분: 입력이 양수면 1, 아니면 0 (out.data>0가 True/False→1/0로 쓰임)
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # 이 노드(보통 손실)에서 시작해 그래프 전체의 기울기를 계산하는 역전파
        topo = []               # 위상 정렬된 노드 목록(자식이 부모보다 앞에 오도록)
        visited = set()         # 이미 방문한 노드 기록(중복 방지)
        def build_topo(v):
            if v not in visited:            # 아직 방문 안 했으면
                visited.add(v)              #   방문 표시
                for child in v._prev:       #   먼저 모든 자식을 재귀적으로 처리
                    build_topo(child)
                topo.append(v)              #   자식들 뒤에 자신을 추가(위상 순서)
        build_topo(self)                    # 그래프를 위상 정렬
        self.grad = 1           # 출력(손실) 자신에 대한 기울기는 1 (∂L/∂L = 1)
        for v in reversed(topo):  # 위상 순서의 역순 = 출력→입력 방향으로 순회
            v._backward()         # 각 노드의 역전파 규칙을 실행해 기울기를 자식에게 전파

    # 아래는 편의를 위한 파이썬 연산자 오버로드(뺄셈·나눗셈·역방향 연산 등을 위 기본 연산으로 표현)
    def __neg__(self): return self * -1                    # -self
    def __radd__(self, other): return self + other         # other + self (숫자 + Value)
    def __sub__(self, other): return self + (-other)       # self - other
    def __rsub__(self, other): return other + (-self)      # other - self
    def __rmul__(self, other): return self * other         # other * self
    def __truediv__(self, other): return self * other**-1  # self / other = self * other^(-1)
    def __rtruediv__(self, other): return other * self**-1 # other / self
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"  # 디버깅 출력 형식

# ---------------------------------------------------------------------------
# 모델 파라미터
# ---------------------------------------------------------------------------
n_embd = 16      # 임베딩 차원(각 토큰을 나타내는 벡터의 길이)
n_head = 4       # 멀티헤드 어텐션의 헤드 수
n_layer = 1      # 트랜스포머 블록(레이어) 수
block_size = 8   # 컨텍스트 길이(한 번에 볼 수 있는 최대 토큰 수)
head_dim = n_embd // n_head  # 헤드 하나가 담당하는 차원 = 16 // 4 = 4
# nout×nin 크기의 가중치 행렬을 만드는 팩토리. 각 원소는 작은 가우시안 난수를 담은 Value.
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# 임베딩·출력 헤드 가중치. wte=토큰 임베딩, wpe=위치 임베딩, lm_head=출력 투영
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):  # 각 레이어마다 어텐션/MLP 가중치를 만든다
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)           # Query 투영 행렬
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)           # Key 투영 행렬
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)           # Value 투영 행렬
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)    # 출력 투영(0에서 시작→잔차 안정화)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)       # MLP 1층: 16→64로 확장
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)  # MLP 2층: 64→16으로 축소(0에서 시작)
# 모든 행렬의 모든 원소(Value)를 하나의 평평한 리스트로 모은다 → 옵티마이저가 순회하기 편하게
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")  # 총 파라미터 수 출력(약 4064개)

# ---------------------------------------------------------------------------
# 모델 아키텍처 (gpt.py와 동일)
# ---------------------------------------------------------------------------
def linear(x, w):
    # 선형 변환 y = Wx: 가중치 행렬 w의 각 행(wo)과 입력 x의 내적을 모아 출력 벡터를 만든다
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 로짓 리스트를 확률 분포(합=1)로 변환
    max_val = max(val.data for val in logits)          # 최댓값(지수 오버플로 방지를 위한 수치 안정화)
    exps = [(val - max_val).exp() for val in logits]   # 각 값에서 최댓값을 빼고 e^x
    total = sum(exps)                                  # 지수들의 합(분모)
    return [e / total for e in exps]                   # 각 지수를 합으로 나눠 확률화

def rmsnorm(x):
    # RMSNorm: 벡터를 제곱평균제곱근으로 나눠 크기를 정규화(LayerNorm보다 단순, 평균 빼기 없음)
    ms = sum(xi * xi for xi in x) / len(x)  # 제곱들의 평균(mean square)
    scale = (ms + 1e-5) ** -0.5             # 1/sqrt(ms). 1e-5는 0으로 나누는 것 방지
    return [xi * scale for xi in x]         # 각 원소에 스케일을 곱함

def gpt(token_id, pos_id, keys, values):
    # 한 토큰 위치에 대한 순전파. keys/values는 이전 위치들의 K·V 캐시(레이어별 리스트).
    tok_emb = state_dict['wte'][token_id]  # 토큰 ID로 토큰 임베딩 벡터를 조회
    pos_emb = state_dict['wpe'][pos_id]    # 위치 ID로 위치 임베딩 벡터를 조회
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 토큰 임베딩 + 위치 임베딩(원소별 덧셈)
    x = rmsnorm(x)                                 # 블록에 들어가기 전 정규화

    for li in range(n_layer):  # 각 트랜스포머 블록을 순서대로 통과
        # --- 1) 멀티헤드 어텐션 블록 ---
        x_residual = x                                       # 잔차 연결을 위해 입력을 보관
        x = rmsnorm(x)                                       # 어텐션 전 정규화(Pre-LN)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])      # Query 투영
        k = linear(x, state_dict[f'layer{li}.attn_wk'])      # Key 투영
        v = linear(x, state_dict[f'layer{li}.attn_wv'])      # Value 투영
        keys[li].append(k)      # 현재 위치의 K를 이 레이어의 캐시에 추가
        values[li].append(v)    # 현재 위치의 V를 이 레이어의 캐시에 추가
        x_attn = []             # 모든 헤드의 출력을 이어 붙일 리스트
        for h in range(n_head):                              # 각 헤드를 독립적으로 계산
            hs = h * head_dim                                # 이 헤드가 차지하는 구간의 시작 인덱스
            q_h = q[hs:hs+head_dim]                           # 현재 위치 Query의 이 헤드 조각
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]    # 지금까지 모든 위치의 Key 조각들
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 지금까지 모든 위치의 Value 조각들
            # 어텐션 점수: 현재 Q와 각 과거 위치 K의 내적을 sqrt(head_dim)으로 나눔(스케일링)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)              # 점수를 확률(가중치)로 변환
            # 각 위치의 Value를 가중치로 가중합 → 이 헤드의 출력 벡터
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)                          # 헤드 출력을 이어 붙임
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo']) # 합쳐진 헤드 출력을 출력 투영
        x = [a + b for a, b in zip(x, x_residual)]           # 잔차 연결: 어텐션 결과 + 원래 입력
        # --- 2) MLP(FFN) 블록 ---
        x_residual = x                                       # 다시 잔차용 입력 보관
        x = rmsnorm(x)                                       # MLP 전 정규화
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])      # 1층: 16→64 확장
        x = [xi.relu() ** 2 for xi in x]                     # 비선형 활성화 ReLU²(ReLU 후 제곱)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])      # 2층: 64→16 축소
        x = [a + b for a, b in zip(x, x_residual)]           # 잔차 연결: MLP 결과 + 입력

    logits = linear(x, state_dict['lm_head'])  # 최종 표현을 어휘 크기(27)의 로짓으로 투영
    return logits                              # 다음 토큰에 대한 점수(로짓) 반환

# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------
# Adam 하이퍼파라미터: 학습률, 1차/2차 모멘트 감쇠계수, 0나눗셈 방지 epsilon
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # 각 파라미터의 1차 모멘트(기울기 이동평균) 버퍼, 0으로 초기화
v = [0.0] * len(params)  # 각 파라미터의 2차 모멘트(기울기 제곱 이동평균) 버퍼, 0으로 초기화

print(f"\nTraining for {num_steps} steps...")  # 학습 시작 알림
for step in range(num_steps):                  # 지정한 스텝 수만큼 반복
    doc = docs[step % len(docs)]  # 문서를 순환하며 하나 선택(스텝이 문서 수를 넘으면 처음부터)
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]  # 이름을 토큰화하고 양 끝에 <BOS>를 붙임
    n = min(block_size, len(tokens) - 1)  # 예측할 위치 수(컨텍스트 길이로 제한)

    # 이 스텝의 KV 캐시를 레이어별 빈 리스트로 초기화
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []  # 각 위치의 손실을 모을 리스트
    for pos_id in range(n):  # 시퀀스의 각 위치에서 다음 토큰을 예측
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 현재 입력 토큰과 정답(다음 토큰)
        logits = gpt(token_id, pos_id, keys, values)  # 순전파로 로짓 계산(그래프도 함께 구축)
        probs = softmax(logits)                       # 로짓을 확률로 변환
        loss_t = -probs[target_id].log()              # 교차 엔트로피: 정답 확률의 음의 로그
        losses.append(loss_t)                         # 위치별 손실 누적
    loss = (1 / n) * sum(losses)  # 시퀀스 전체 위치의 평균 손실

    loss.backward()  # 역전파: 손실에 대한 모든 파라미터의 기울기를 그래프를 따라 계산

    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감소(후반으로 갈수록 보폭 축소)
    for i, p in enumerate(params):  # 모든 파라미터에 대해 Adam 갱신 수행
        # 1차 모멘트: 과거 평균과 현재 기울기를 beta1 비율로 섞음(모멘텀 역할)
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 2차 모멘트: 과거 평균과 현재 기울기의 제곱을 beta2 비율로 섞음(적응적 스케일)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))  # 초기 0 편향 보정(1차)
        v_hat = v[i] / (1 - beta2 ** (step + 1))  # 초기 0 편향 보정(2차)
        # 최종 갱신: 보정된 1차 모멘트를 2차 모멘트의 제곱근으로 나눠 파라미터를 조정
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 다음 스텝을 위해 기울기를 0으로 초기화(누적 방지)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")  # 스텝별 손실 출력

# ---------------------------------------------------------------------------
# 학습된 모델 저장 (이 스크립트의 고유 부분)
# ---------------------------------------------------------------------------
model = {  # 저장할 내용을 하나의 딕셔너리로 조립
    # 하이퍼파라미터 -- 로드 시 아키텍처를 재구성하는 데 필요
    'config': {
        'n_embd': n_embd,          # 임베딩 차원
        'n_head': n_head,          # 헤드 수
        'n_layer': n_layer,        # 레이어 수
        'block_size': block_size,  # 컨텍스트 길이
        'vocab_size': vocab_size,  # 어휘 크기
    },
    # 토크나이저 매핑 -- 텍스트 인코딩/디코딩에 필요(가중치와 반드시 함께 저장)
    'chars': chars,
    # 학습된 가중치 -- 모델의 실제 지식. Value 래퍼를 벗기고 순수 float(p.data)만 저장
    'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
}

with open(output_path, 'w') as f:  # 저장 경로를 쓰기 모드로 열고(블록 종료 시 자동 close)
    json.dump(model, f)            # 딕셔너리를 JSON 텍스트로 직렬화해 파일에 기록

size_kb = os.path.getsize(output_path) / 1024  # 저장된 파일 크기를 KB 단위로 계산
print(f"\nModel saved to {output_path} ({size_kb:.1f} KB)")  # 저장 완료 및 크기 출력
print(f"Contains: config, tokenizer ({vocab_size} tokens), weights ({len(params)} params)")  # 내용 요약
