"""
[이 파일은 무엇인가요?]
이 프로그램은 '이름 짓는 인공지능(GPT)'을 컴퓨터에게 가르치고(=학습),
가르친 결과(숫자 묶음)를 파일로 저장하는 프로그램입니다.

- GPT: 앞의 글자들을 보고 '다음 글자'를 계속 이어 쓰는 방식으로 글을 만드는 인공지능입니다.
        여기서는 사람 이름 3만여 개를 보고, 그럴듯한 새 이름을 만들도록 배웁니다.
- 학습: 처음엔 아무렇게나 대답하지만, 정답과 얼마나 틀렸는지를 계산해
        내부 숫자들을 조금씩 고쳐 점점 똑똑해지는 과정입니다.

이 주석들은 파이썬(프로그래밍 언어) 문법과 GPT를 '전혀 모른다'고 가정하고,
각 코드 줄 바로 아래에 그 줄이 무슨 일을 하는지 쉬운 말로 설명합니다.
코드 자체는 원본 train.py와 완전히 같습니다.

사용법: uv run train.py [--steps 500] [--output model.json]
"""

# 'import'는 "다른 사람이 미리 만들어 둔 도구 상자를 가져와 쓰겠다"는 뜻입니다.
import os
# os: 컴퓨터의 '파일'을 다루는 도구 상자(파일이 있는지 확인, 파일 크기 재기 등)
import sys
# sys: 프로그램 실행과 관련된 도구(사용자가 입력한 옵션 읽기, 프로그램 끝내기 등)
import math
# math: 수학 계산 도구(로그, 지수 같은 함수)
import json
# json: 숫자·글자 묶음을 '텍스트 파일'로 저장하고 읽는 표준 형식 도구
import random
# random: 무작위(랜덤) 숫자를 만드는 도구

# ---------------------------------------------------------------------------
# 사용자가 입력한 옵션 읽기
# ---------------------------------------------------------------------------
# '=' 는 오른쪽 값에 왼쪽 이름표를 붙이는 것입니다(수학의 등호가 아니라 '저장'입니다).
num_steps = 500
# num_steps 라는 이름에 500을 저장 → 몇 번 반복해 배울지의 기본값(기본 500번)
output_path = 'model.json'
# 따옴표로 감싼 것은 '글자(문자열)'입니다. 결과를 저장할 파일 이름의 기본값
args = sys.argv[1:]
# 사용자가 프로그램 뒤에 붙여 입력한 옵션들의 목록. [1:]은 맨 앞(프로그램 이름)을 뺀 나머지
i = 0
# i 라는 이름에 0을 저장. 목록을 앞에서부터 하나씩 짚어갈 '순번'으로 씁니다
while i < len(args):
    # 'while ...:' 은 조건이 참인 동안 아래 들여쓴 줄을 반복합니다. len(args)는 목록의 개수
    if args[i] == '--steps' and i + 1 < len(args):
        # 'if ...:' 은 조건이 참이면 아래를 실행. '=='는 '값이 같은가?' 비교. 'and'는 '그리고'
        num_steps = int(args[i + 1]); i += 2
        # int(...)는 글자를 숫자로 바꿈. 세미콜론(;)은 한 줄에 두 명령. i += 2는 i를 2 늘림
    elif args[i] == '--output' and i + 1 < len(args):
        # 'elif'는 "앞의 if가 아니라면, 이 조건이면" 이라는 뜻(else if의 줄임)
        output_path = args[i + 1]; i += 2
        # '--output' 다음에 온 글자를 저장 경로로 삼고, 순번을 2 늘림
    else:
        # 'else'는 "위의 어떤 조건에도 안 맞으면"
        print(f"Unknown arg: {args[i]}"); sys.exit(1)
        # print(...)는 화면에 출력. f"...{값}..."는 중괄호 안 값을 글자에 끼워 넣는 방식.
        # sys.exit(1)은 "문제가 있어 프로그램을 끝냄"(1은 '오류'라는 신호)

# ---------------------------------------------------------------------------
# 배울 데이터(이름 목록) 준비하기
# ---------------------------------------------------------------------------
random.seed(42)
# 무작위의 '출발점'을 42로 고정 → 매번 실행해도 똑같은 결과가 나오게 함(재현성)
if not os.path.exists('input.txt'):
    # 'not'은 '아니면'. input.txt(이름 데이터 파일)가 '없으면' 아래를 실행
    import urllib.request
    # 인터넷에서 파일을 내려받는 도구(여기서만 필요해서 이 자리에서 가져옴)
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    # 내려받을 이름 데이터(약 3만 2천 개)의 인터넷 주소
    urllib.request.urlretrieve(names_url, 'input.txt')
    # 그 주소의 내용을 input.txt 라는 파일로 저장
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
# 대괄호[]는 여러 값을 담는 '목록(리스트)'. 이 문법은 "파일을 줄 단위로 나눠, 빈 줄은 빼고,
# 각 줄의 양옆 공백을 없앤 이름들"을 모은 목록을 만듭니다. docs = 이름 3만여 개의 목록
random.shuffle(docs)
# 목록의 순서를 무작위로 뒤섞음(특정 순서에 치우쳐 배우지 않도록)
print(f"num docs: {len(docs)}")
# 이름이 몇 개인지 화면에 출력(len은 개수)

# ---------------------------------------------------------------------------
# 글자를 숫자로 바꾸는 '사전' 만들기 (컴퓨터는 글자를 숫자로만 다룹니다)
# ---------------------------------------------------------------------------
chars = ['<BOS>'] + sorted(set(''.join(docs)))
# ''.join(docs)는 모든 이름을 한 줄로 이어 붙임. set(...)은 중복을 없앤 '고유 글자 모음'.
# sorted(...)는 정렬. 맨 앞의 '<BOS>'는 '이름 시작/끝'을 뜻하는 특별한 표식.
# chars = 사용 글자 목록(예: <BOS>, a, b, ... , z)
vocab_size = len(chars)
# 글자 종류의 개수(여기서는 27개 = 알파벳 26 + 특별표식 1)
stoi = { ch:i for i, ch in enumerate(chars) }
# 중괄호{}는 '이름표→값'을 짝지어 담는 '사전(딕셔너리)'. enumerate는 (순번, 값)을 함께 줌.
# stoi = '글자 → 숫자' 사전(예: 'a' → 1). string-to-integer의 줄임
itos = { i:ch for i, ch in enumerate(chars) }
# itos = '숫자 → 글자' 사전(위의 반대 방향). integer-to-string의 줄임
BOS = stoi['<BOS>']
# 특별표식 '<BOS>'에 해당하는 숫자를 미리 꺼내 BOS 라는 이름에 저장
print(f"vocab size: {vocab_size}")
# 글자 종류 개수를 화면에 출력

# ---------------------------------------------------------------------------
# '자동 미분 엔진' 만들기
#   - 학습이란 '내부 숫자를 어느 방향으로 바꿔야 정답에 가까워지는지'를 알아내는 일입니다.
#   - 그 '방향'을 자동으로 계산해 주는 장치가 아래의 Value 입니다.
# ---------------------------------------------------------------------------
class Value:
    # 'class'는 '새로운 종류의 물건'을 정의하는 설계도입니다. 여기서는 '숫자 하나'를 담는 물건.
    """하나의 스칼라 값과 그 기울기를 저장한다."""
    # 이 물건(Value)은 '값' 하나와, '그 값을 어느 쪽으로 바꿔야 하는지(기울기)'를 함께 가집니다.

    def __init__(self, data, _children=(), _op=''):
        # 'def'는 '함수(작업 묶음)'를 정의. __init__은 물건을 처음 만들 때 자동 실행되는 준비 작업.
        # 'self'는 '지금 만들어지는 그 물건 자신'을 가리킵니다.
        self.data = data
        # 이 물건이 담는 '실제 숫자 값'
        self.grad = 0
        # '기울기': 이 값을 조금 바꾸면 정답과의 오차가 얼마나 변하는지. 처음엔 0
        self._backward = lambda: None
        # 'lambda'는 이름 없는 짧은 함수. 여기선 '아무 일도 안 하는 함수'가 기본값
        self._prev = set(_children)
        # 이 값을 만들어 낸 '재료 값들'의 모음(나중에 기울기를 거꾸로 전달할 때 씀)
        self._op = _op
        # 이 값이 어떤 계산(+, * 등)으로 생겼는지 기록(설명·디버깅용)

    def __add__(self, other):
        # 이름이 __add__ 이면 '+' 기호가 이 물건에 어떻게 동작할지 정하는 것입니다(더하기).
        other = other if isinstance(other, Value) else Value(other)
        # 상대가 그냥 숫자면 Value 물건으로 감쌈('A if 조건 else B'는 조건이면 A, 아니면 B)
        out = Value(self.data + other.data, (self, other), '+')
        # 두 값을 더한 '새 결과 물건'을 만듦(재료로 self와 other를 기억)
        def _backward():
            # 이 더하기의 '기울기 되돌리기' 규칙
            self.grad += out.grad
            # '+= '는 '기존 값에 더해서 다시 저장'. 더하기에서는 오차 책임을 그대로 전달
            other.grad += out.grad
            # 반대쪽 재료에도 똑같이 그대로 전달
        out._backward = _backward
        # 결과 물건에 위 규칙을 붙여 둠
        return out
        # 'return'은 계산 결과를 돌려줌. 더한 결과 물건을 반환

    def __mul__(self, other):
        # __mul__ 이면 '*' 기호(곱하기)가 어떻게 동작할지 정의
        other = other if isinstance(other, Value) else Value(other)
        # 상대가 숫자면 Value로 감쌈
        out = Value(self.data * other.data, (self, other), '*')
        # 두 값을 곱한 결과 물건을 만듦
        def _backward():
            # 곱하기의 기울기 규칙
            self.grad += other.data * out.grad
            # 곱하기에서는 '상대편 값'을 곱해서 오차 책임을 전달
            other.grad += self.data * out.grad
            # 반대쪽도 마찬가지(상대편 값을 곱함)
        out._backward = _backward
        # 규칙 붙이기
        return out
        # 결과 반환

    def __pow__(self, other):
        # __pow__ 이면 '**' 기호(거듭제곱)가 어떻게 동작할지 정의
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # 'assert'는 조건이 참인지 확인(아니면 멈춤). 지수는 보통 숫자만 허용
        out = Value(self.data**other, (self,), f'**{other}')
        # 값을 거듭제곱한 결과 물건을 만듦('**'가 거듭제곱)
        def _backward():
            # 거듭제곱의 기울기 규칙
            self.grad += (other * self.data**(other-1)) * out.grad
            # 수학의 미분 공식(n제곱의 변화율은 n×(n-1제곱))을 적용
        out._backward = _backward
        # 규칙 붙이기
        return out
        # 결과 반환

    def log(self):
        # 'log'는 로그 계산을 하는 작업(수학 함수)
        out = Value(math.log(self.data), (self,), 'log')
        # 값에 로그를 취한 결과 물건을 만듦
        def _backward():
            # 로그의 기울기 규칙
            self.grad += (1 / self.data) * out.grad
            # 로그의 변화율은 1을 원래 값으로 나눈 것
        out._backward = _backward
        # 규칙 붙이기
        return out
        # 결과 반환

    def exp(self):
        # 'exp'는 지수 계산(e를 그 값만큼 거듭제곱)
        out = Value(math.exp(self.data), (self,), 'exp')
        # 지수 계산 결과 물건을 만듦
        def _backward():
            # 지수의 기울기 규칙
            self.grad += out.data * out.grad
            # 지수의 변화율은 '자기 자신'과 같으므로 결과값을 그대로 곱함
        out._backward = _backward
        # 규칙 붙이기
        return out
        # 결과 반환

    def relu(self):
        # 'relu'는 '음수는 0으로, 양수는 그대로' 만드는 아주 흔한 처리(비선형이라 부름)
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        # 값이 0보다 작으면 0, 아니면 그대로인 결과 물건을 만듦
        def _backward():
            # relu의 기울기 규칙
            self.grad += (out.data > 0) * out.grad
            # 결과가 양수일 때만 책임을 통과시키고, 아니면 0(True/False가 1/0으로 쓰임)
        out._backward = _backward
        # 규칙 붙이기
        return out
        # 결과 반환

    def backward(self):
        # 이 물건(보통 '오차')에서 시작해, 모든 재료 값의 기울기를 한꺼번에 계산하는 작업
        topo = []
        # 빈 목록. 계산 순서를 정리해 담을 곳
        visited = set()
        # 이미 들른 물건을 기록(같은 것을 두 번 처리하지 않도록)
        def build_topo(v):
            # 물건들의 연결을 따라가며 '계산 순서'를 만드는 도우미 함수
            if v not in visited:
                # 아직 안 들른 물건이면
                visited.add(v)
                # 들렀다고 표시
                for child in v._prev:
                    # 'for ... in ...:'은 모음의 원소를 하나씩 꺼내 반복. 이 물건의 재료들을 순회
                    build_topo(child)
                    # 각 재료를 먼저 처리(재귀: 함수가 자기 자신을 다시 부름)
                topo.append(v)
                # 재료를 다 넣은 뒤 자신을 목록에 추가
        build_topo(self)
        # 이 물건을 시작점으로 순서를 만듦
        self.grad = 1
        # 출발점(오차) 자신의 기울기는 1로 둠(자기 자신에 대한 변화율은 1)
        for v in reversed(topo):
            # reversed(...)는 순서를 거꾸로. 결과쪽에서 재료쪽으로 되짚어 감
            v._backward()
            # 각 물건의 '기울기 되돌리기' 규칙을 실행

    # 아래는 '-','-','/' 같은 다른 기호들도 이 물건에서 동작하도록 짧게 정의한 것들입니다.
    def __neg__(self): return self * -1
    # 앞에 붙는 마이너스(-self)는 -1을 곱한 것과 같게 처리
    def __radd__(self, other): return self + other
    # '숫자 + 물건' 순서로 써도 되게 함
    def __sub__(self, other): return self + (-other)
    # 빼기(a - b)는 'a + (-b)'로 처리
    def __rsub__(self, other): return other + (-self)
    # '숫자 - 물건' 순서를 지원
    def __rmul__(self, other): return self * other
    # '숫자 * 물건' 순서를 지원
    def __truediv__(self, other): return self * other**-1
    # 나누기(a / b)는 'a 곱하기 b의 -1제곱'으로 처리
    def __rtruediv__(self, other): return other * self**-1
    # '숫자 / 물건' 순서를 지원
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"
    # 이 물건을 화면에 찍을 때 보여줄 글자 모양

# ---------------------------------------------------------------------------
# 모델의 '내부 숫자(가중치)' 준비하기
#   - 가중치: 학습으로 조금씩 바꿔 나갈 숫자들. 이 숫자들 안에 '배운 지식'이 담깁니다.
# ---------------------------------------------------------------------------
n_embd = 16
# 글자 하나를 표현하는 숫자의 개수(길이 16짜리 숫자 묶음으로 한 글자를 나타냄)
n_head = 4
# '어텐션'을 몇 갈래로 나눠 볼지(뒤에서 설명). 여기선 4갈래
n_layer = 1
# 계산 블록을 몇 층 쌓을지. 여기선 1층
block_size = 8
# 한 번에 볼 수 있는 최대 글자 수(문맥 길이)
head_dim = n_embd // n_head
# '//'는 나눗셈의 몫(소수점 버림). 한 갈래가 맡는 숫자 개수 = 16 ÷ 4 = 4
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# 'lambda'로 만든 짧은 함수. nout×nin 크기의 '숫자 표(행렬)'를 만들되, 각 칸을 작은 무작위
# 숫자로 채웁니다. random.gauss는 0 근처의 무작위 값. (밑줄 _ 은 '안 쓰는 값'이라는 관습)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
# 여러 숫자 표를 이름표와 함께 사전에 담음.
# wte=글자를 숫자묶음으로 바꾸는 표, wpe='몇 번째 위치인지'를 나타내는 표, lm_head=마지막 예측용 표
for i in range(n_layer):
    # 층(레이어) 수만큼 반복하며 각 층의 계산용 표들을 만듦. range(n)은 0..n-1
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    # 'Query(질문)' 표 — 어텐션에서 "나는 무엇을 찾는가"를 만드는 데 씀
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    # 'Key(열쇠)' 표 — "나는 무슨 정보를 가졌는가"를 만드는 데 씀
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    # 'Value(값)' 표 — 실제로 전달할 정보를 만드는 데 씀
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    # 어텐션 결과를 정리해 내보내는 표(처음엔 0으로 시작해 학습을 안정화)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    # 정보를 넓게 펼치는 표(16개 → 64개로 확장)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)
    # 다시 좁히는 표(64개 → 16개로 축소, 처음엔 0으로 시작)
params = [p for mat in state_dict.values() for row in mat for p in row]
# 모든 표의 모든 칸(숫자)을 하나의 긴 목록으로 모음(학습할 때 하나씩 돌기 편하게)
print(f"num params: {len(params)}")
# 조정할 숫자가 총 몇 개인지 출력(약 4064개)

# ---------------------------------------------------------------------------
# 모델의 '계산 방법' 정의하기 (입력 글자 → 다음 글자 예측)
# ---------------------------------------------------------------------------
def linear(x, w):
    # 'linear'는 숫자묶음 x를 표 w에 통과시키는 기본 계산(각 칸을 곱해 더함)
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    # zip은 두 목록을 짝지어 줌. 표의 각 줄과 입력을 곱해 더한 결과 목록을 돌려줌

def softmax(logits):
    # 'softmax'는 여러 점수를 '합이 1인 확률'로 바꿔 줌(예: 각 글자가 다음에 올 확률)
    max_val = max(val.data for val in logits)
    # 가장 큰 점수를 구함(계산이 너무 커지는 것을 막는 안전장치)
    exps = [(val - max_val).exp() for val in logits]
    # 각 점수에서 최댓값을 뺀 뒤 지수(exp) 계산 → 모두 양수로 만듦
    total = sum(exps)
    # 그 값들의 합계
    return [e / total for e in exps]
    # 각 값을 합계로 나눠 '확률'로 만듦(전부 더하면 1)

def rmsnorm(x):
    # 'rmsnorm'은 숫자묶음의 크기를 일정하게 맞춰 계산을 안정시키는 처리
    ms = sum(xi * xi for xi in x) / len(x)
    # 각 숫자를 제곱해 평균낸 값(크기의 척도)
    scale = (ms + 1e-5) ** -0.5
    # 그 크기의 제곱근으로 나눌 배율을 구함(1e-5는 0으로 나눠 터지는 것 방지)
    return [xi * scale for xi in x]
    # 모든 숫자에 배율을 곱해 크기를 맞춤

def gpt(token_id, pos_id, keys, values):
    # 이 함수가 GPT의 핵심: '현재 글자(token_id)와 위치(pos_id)'를 받아 '다음 글자 점수'를 냄.
    # keys/values는 앞서 본 글자들의 정보를 저장해 둔 메모(다시 계산 안 하려고)
    tok_emb = state_dict['wte'][token_id]
    # 현재 글자를 '의미 숫자묶음'으로 바꿈(글자 임베딩)
    pos_emb = state_dict['wpe'][pos_id]
    # 현재 '위치'도 숫자묶음으로 바꿈(몇 번째 글자인지 알려 줌)
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    # 글자 정보와 위치 정보를 자리마다 더해 하나로 합침
    x = rmsnorm(x)
    # 크기를 맞춰 정리

    for li in range(n_layer):
        # 계산 블록(층)을 순서대로 통과
        x_residual = x
        # 나중에 다시 더하려고 지금 입력을 따로 보관(이걸 '잔차 연결'이라 함)
        x = rmsnorm(x)
        # 어텐션 전에 크기 정리
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        # '질문(Q)' 만들기
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        # '열쇠(K)' 만들기
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        # '값(V)' 만들기
        keys[li].append(k)
        # 지금 만든 열쇠를 메모에 추가(append는 목록 맨 뒤에 붙이기)
        values[li].append(v)
        # 지금 만든 값을 메모에 추가
        x_attn = []
        # 각 갈래(헤드) 결과를 이어 붙일 빈 목록
        for h in range(n_head):
            # 어텐션을 여러 갈래로 나눠 각각 계산
            hs = h * head_dim
            # 이 갈래가 맡은 부분의 시작 위치
            q_h = q[hs:hs+head_dim]
            # [시작:끝]은 '그 구간만 잘라내기(슬라이싱)'. 이 갈래의 질문 부분
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            # 지금까지 본 모든 글자의 '열쇠' 중 이 갈래 부분들
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # 지금까지 본 모든 글자의 '값' 중 이 갈래 부분들
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            # 내 질문과 각 글자의 열쇠가 얼마나 잘 맞는지 점수를 계산(맞을수록 높음)
            attn_weights = softmax(attn_logits)
            # 그 점수를 '확률(주목 정도)'로 바꿈
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            # 주목 정도만큼 각 글자의 '값'을 섞어 이 갈래의 결과를 만듦
            x_attn.extend(head_out)
            # extend는 목록을 통째로 이어 붙임. 갈래 결과를 전체 결과에 추가
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        # 여러 갈래 결과를 정리해 내보냄
        x = [a + b for a, b in zip(x, x_residual)]
        # 아까 보관해 둔 입력을 다시 더함(잔차 연결: 정보 손실을 막음)
        x_residual = x
        # 다음 단계(MLP)를 위해 또 보관
        x = rmsnorm(x)
        # 크기 정리
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        # 정보를 넓게 펼침(16 → 64)
        x = [xi.relu() ** 2 for xi in x]
        # 음수는 0으로 만든 뒤 제곱(비선형 처리: 복잡한 규칙을 배우게 해 줌)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        # 다시 좁힘(64 → 16)
        x = [a + b for a, b in zip(x, x_residual)]
        # 또 한 번 잔차 연결(입력을 더함)

    logits = linear(x, state_dict['lm_head'])
    # 최종 숫자묶음을 '글자마다의 점수(27개)'로 바꿈
    return logits
    # 다음 글자 점수를 돌려줌(높을수록 그 글자가 올 가능성이 큼)

# ---------------------------------------------------------------------------
# 실제로 '학습'시키기 (틀린 만큼 내부 숫자를 조금씩 고치기)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
# 한 줄에 여러 이름에 값을 나눠 저장. learning_rate=한 번에 얼마나 고칠지(보폭),
# 나머지는 'Adam'이라는 똑똑한 조정 방법이 쓰는 설정값들(1e-2는 0.01을 뜻함)
m = [0.0] * len(params)
# [0.0]*개수는 '0.0을 개수만큼 채운 목록'. 각 숫자의 '조정 기록1'을 담아 둘 곳
v = [0.0] * len(params)
# 각 숫자의 '조정 기록2'를 담아 둘 곳
print(f"\nTraining for {num_steps} steps...")
# 학습을 시작한다고 알림(\n은 줄바꿈)
for step in range(num_steps):
    # 정해진 횟수만큼 학습을 반복
    doc = docs[step % len(docs)]
    # '%'는 나눈 나머지. 이름 목록을 돌아가며 하나씩 꺼내 씀
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
    # 이름의 각 글자를 숫자로 바꾸고, 앞뒤에 시작/끝 표식을 붙인 '숫자 목록'을 만듦
    n = min(block_size, len(tokens) - 1)
    # 이번에 예측을 연습할 글자 수(문맥 길이를 넘지 않게 제한)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    # 앞서 본 글자 정보를 저장할 '메모'를 층마다 빈 목록으로 새로 준비
    losses = []
    # 각 위치에서 얼마나 틀렸는지를 모을 목록
    for pos_id in range(n):
        # 이름의 각 위치에서 '다음 글자'를 맞히는 연습
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # 지금 글자(token_id)와, 맞혀야 할 정답 글자(target_id: 바로 다음 글자)
        logits = gpt(token_id, pos_id, keys, values)
        # 모델에게 다음 글자 점수를 예측하게 함
        probs = softmax(logits)
        # 점수를 확률로 바꿈
        loss_t = -probs[target_id].log()
        # '틀린 정도(손실)': 정답 글자의 확률이 낮을수록 이 값이 커짐
        losses.append(loss_t)
        # 이 위치의 손실을 목록에 추가
    loss = (1 / n) * sum(losses)
    # 이름 전체 위치의 손실 평균(이 값을 줄이는 게 학습의 목표)
    loss.backward()
    # 앞에서 만든 자동 미분으로, 각 내부 숫자를 '어느 쪽으로 바꿔야 손실이 줄지' 계산
    lr_t = learning_rate * (1 - step / num_steps)
    # 보폭을 뒤로 갈수록 조금씩 줄임(처음엔 크게, 나중엔 세밀하게 고침)
    for i, p in enumerate(params):
        # 모든 내부 숫자(p)를 하나씩 꺼내 조정. enumerate는 순번 i도 함께 줌
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 최근 '바꿀 방향'들을 부드럽게 평균냄(들쭉날쭉함을 줄임)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 방향의 '크기'도 추적해서 숫자마다 보폭을 알맞게 조절
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        # 학습 초반의 치우침을 보정(1)
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # 학습 초반의 치우침을 보정(2)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # '-='는 빼서 다시 저장. 계산한 방향·보폭만큼 이 숫자를 실제로 조금 고침
        p.grad = 0
        # 다음 번 계산을 위해 방향값을 0으로 비움
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
    # 몇 번째 학습인지와 현재 틀린 정도(손실)를 출력(숫자가 점점 줄어들면 잘 배우는 것)

# ---------------------------------------------------------------------------
# 다 배운 모델을 파일로 저장하기 (이 프로그램만의 특별한 부분)
# ---------------------------------------------------------------------------
model = {
    # 저장할 내용을 하나의 사전(딕셔너리)으로 묶음
    # 아래 config는 '모델의 설정값'. 나중에 불러올 때 똑같은 구조를 다시 만들기 위해 필요
    'config': {
        'n_embd': n_embd,
        # 글자 하나를 나타내는 숫자 개수
        'n_head': n_head,
        # 어텐션 갈래 수
        'n_layer': n_layer,
        # 층 수
        'block_size': block_size,
        # 문맥 길이
        'vocab_size': vocab_size,
        # 글자 종류 수
    },
    # chars는 '글자 목록'. 저장한 숫자를 다시 글자로 해석하려면 반드시 함께 있어야 함
    'chars': chars,
    # weights는 '배운 내부 숫자들'. 계산장치(Value)를 벗기고 '순수한 숫자'만 꺼내 저장
    'weights': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
}
with open(output_path, 'w') as f:
    # 'with open(...)'은 파일을 열고, 이 블록이 끝나면 자동으로 닫아 줌. 'w'는 쓰기 모드
    json.dump(model, f)
    # 위에서 묶은 model을 JSON 형식의 글자로 바꿔 파일에 기록
size_kb = os.path.getsize(output_path) / 1024
# 저장된 파일의 크기를 KB(킬로바이트) 단위로 계산
print(f"\nModel saved to {output_path} ({size_kb:.1f} KB)")
# 어디에 얼마 크기로 저장했는지 알림
print(f"Contains: config, tokenizer ({vocab_size} tokens), weights ({len(params)} params)")
# 저장한 내용 요약을 알림(설정, 글자표, 배운 숫자들)
