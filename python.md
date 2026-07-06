# Python 키워드 정리

이 문서는 이 저장소의 `*.py` 파일들(`gpt.py`, `persistence/train.py`, `persistence/run.py`, `model-sources/download_wikipedia.py`)에서 **실제로 사용된 Python 키워드**들의 역할을 상세히 정리한 것입니다. 각 키워드마다 역할 설명과 함께, 저장소 코드에서 그대로 가져온 실제 사용 예를 담았습니다.

> **키워드(keyword)란?** Python 언어에 미리 예약된 단어로, 변수 이름 등으로 쓸 수 없습니다. 아래 24개는 이 저장소의 `.py` 파일에서 사용된 것들입니다. (`or`, `pass`, `continue`, `raise`, `finally`, `global`, `del`, `yield`, `async`/`await` 등은 이 코드에서는 쓰이지 않았습니다.)

---

## 1. 상수 값 (Constant Values)

### `True` / `False`
불리언(참/거짓) 값을 나타내는 상수. 조건 판단과 플래그에 쓰입니다.

```python
dates.sort(reverse=True)   # download_wikipedia.py:74 — 내림차순 정렬 옵션
return False               # download_wikipedia.py:162 — 함수가 "실패/거짓"을 반환
```

### `None`
"값이 없음"을 나타내는 특별한 객체. 기본값이나 "아직 정해지지 않음"을 표현할 때 쓰입니다.

```python
self._backward = lambda: None   # gpt.py:41 — 아무것도 하지 않는 기본 역전파 함수
if dump_date is None:           # download_wikipedia.py:217 — 날짜가 지정되지 않았는지 확인
```

---

## 2. 함수와 클래스 정의 (Functions & Classes)

### `def`
**함수(또는 메서드)를 정의**합니다. 코드를 재사용 가능한 단위로 묶습니다.

```python
def __init__(self, data, _children=(), _op=''):   # gpt.py:38 — Value 클래스의 생성자
```

### `class`
**클래스(객체의 설계도)를 정의**합니다. microgpt의 핵심인 자동 미분 엔진 `Value`가 클래스로 구현됩니다.

```python
class Value:   # gpt.py:35 — 스칼라 값 하나를 감싸는 자동 미분 노드
```

### `return`
함수의 **결과값을 돌려주고** 함수를 종료합니다.

```python
return out   # gpt.py:52 — 연산 결과 Value 객체를 반환
```

### `lambda`
이름 없는 **한 줄짜리 익명 함수**를 만듭니다. 짧은 함수를 즉석에서 정의할 때 유용합니다.

**문법:** `lambda 인자들: 표현식`
- `def`와 달리 **이름이 없고**, 본문은 **단 하나의 표현식**만 올 수 있습니다(문(statement)·여러 줄 불가).
- 그 표현식의 결과가 `return` 없이 **자동으로 반환**됩니다.

```python
lambda x: x + 1        # def f(x): return x + 1 과 동일
lambda: None           # 인자 없음, 항상 None 반환
lambda a, b=2: a * b   # 기본값 인자도 가능
```

이 저장소에는 세 가지 대표적 용례가 있습니다.

**① 인자도 동작도 없는 기본 함수 (gpt.py:41)**
```python
self._backward = lambda: None   # 아무것도 하지 않는 기본 역전파 함수
```
`Value` 노드가 처음 만들어질 때의 기본 `_backward`입니다. 잎(leaf) 노드처럼 역전파할 것이 없을 때, 호출해도 안전하도록 "아무 일도 안 하는 함수"를 넣어 둡니다. 이후 실제 연산에서 진짜 `_backward`로 교체됩니다.

**② 인자와 기본값을 가진 팩토리 함수 (gpt.py:123)**
```python
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), ...}
```
`nout × nin` 크기의 가중치 행렬을 만드는 짧은 팩토리입니다. 인자 `nout`, `nin`과 **기본값 인자** `std=0.02`를 받아, `matrix(vocab_size, n_embd)`나 `matrix(n_embd, n_embd, std=0)`처럼 재사용합니다. `def`로 써도 되지만, 한 줄이라 lambda가 간결합니다.

**③ 다른 함수의 인자로 넘기기 (download_wikipedia.py:177)**
```python
for chunk in iter(lambda: f.read(8192), b''):
    md5_hash.update(chunk)
```
`iter(호출가능객체, 종료값)` 형태에서, **매번 호출될 함수**로 lambda를 넘깁니다. `f.read(8192)`를 반복 호출하다가 결과가 `b''`(빈 바이트, 파일 끝)가 되면 멈춥니다. 이처럼 "호출할 때마다 값을 만드는 짧은 함수"가 필요한 자리에 lambda가 잘 맞습니다.

> **lambda vs def**: lambda는 *식(expression)*이라 변수 대입·인자 전달 등 값이 필요한 자리에 바로 쓸 수 있습니다. 반면 여러 줄 로직, 문서화 문자열(docstring), 복잡한 분기가 필요하면 `def`가 적합합니다.

---

## 3. 조건 분기 (Conditionals)

### `if`
**조건이 참일 때만** 코드 블록을 실행합니다.

```python
if not os.path.exists('input.txt'):   # gpt.py:17 — 데이터 파일이 없으면 다운로드
```

### `elif`
앞의 `if`가 거짓일 때 **다음 조건을 검사**합니다("else if"의 줄임).

```python
elif args[i] == '--samples' and i + 1 < len(args):   # run.py:25 — 다른 CLI 인자 처리
```

### `else`
앞의 모든 조건이 거짓일 때 실행됩니다. 삼항 표현식(`A if 조건 else B`)에도 쓰입니다.

```python
other = other if isinstance(other, Value) else Value(other)   # gpt.py:46 — 아니면 Value로 감쌈
```

---

## 4. 반복 (Loops)

### `for`
**시퀀스(리스트, 문자열 등)의 각 요소를 순회**합니다. 리스트 컴프리헨션 안에서도 쓰입니다.

```python
docs = [l.strip() for l in open('input.txt')...if l.strip()]   # gpt.py:21 — 각 줄을 순회하며 처리
```

### `while`
**조건이 참인 동안 반복**합니다. 반복 횟수가 미리 정해지지 않은 경우에 씁니다.

```python
while i < len(args):   # run.py:22 — CLI 인자를 끝까지 하나씩 처리
```

### `break`
현재 반복문을 **즉시 빠져나옵니다**.

```python
break   # gpt.py:241 — <BOS> 토큰이 생성되면 이름 생성 루프 종료
```

---

## 5. 논리·비교 연산자 (Logic & Comparison)

### `and`
두 조건이 **모두 참일 때만** 참. 왼쪽이 거짓이면 오른쪽은 평가하지 않습니다(단축 평가).

```python
if args[i] == '--model' and i + 1 < len(args):   # run.py:23 — 플래그이면서 뒤에 값이 있는지
```

### `not`
불리언 값을 **뒤집습니다**(참↔거짓).

```python
if not os.path.exists('input.txt'):   # gpt.py:17 — 파일이 "존재하지 않으면"
```

### `in`
값이 **시퀀스 안에 포함되는지** 확인하거나, `for` 문에서 순회 대상을 지정합니다.

```python
for l in open('input.txt')...   # gpt.py:21 — 파일의 각 줄을 대상으로 순회
```

### `is`
두 객체가 **정확히 같은 객체인지**(동일성) 확인합니다. 특히 `None` 비교에 관용적으로 씁니다(`== None`보다 권장).

```python
if dump_date is None:   # download_wikipedia.py:217 — 값이 None 그 자체인지 확인
```

---

## 6. 예외 처리 (Exception Handling)

### `try`
**오류가 날 수 있는 코드**를 감싸, 예외가 발생해도 프로그램이 멈추지 않게 합니다.

```python
try:   # download_wikipedia.py:58 — 네트워크 요청 등 실패 가능성이 있는 작업 시도
```

### `except`
`try` 블록에서 **예외가 발생했을 때 처리**할 코드를 지정합니다. `as`와 함께 예외 객체를 받을 수 있습니다.

```python
except urllib.error.URLError as e:   # download_wikipedia.py:61 — 네트워크 오류를 잡아 처리
```

---

## 7. 검증 (Assertion)

### `assert`
**조건이 참임을 단언**하고, 거짓이면 `AssertionError`를 냅니다. 코드의 전제 조건을 검사하는 방어적 프로그래밍에 쓰입니다.

```python
assert isinstance(other, (int, float)), "only supporting int/float powers for now"   # gpt.py:64
```

---

## 8. 컨텍스트 관리와 별칭 (Context & Aliasing)

### `with`
**컨텍스트 매니저(context manager)**를 사용해, 블록에 들어갈 때 자원을 열고 **블록이 끝나면 자동으로 정리**합니다. 파일·네트워크 연결처럼 "열었으면 반드시 닫아야 하는" 자원에 씁니다.

**동작 원리:** `with 객체 as 이름:` 형태에서, 객체의 `__enter__()`가 블록 진입 시 호출되어 `이름`에 결과를 넣고, 블록을 벗어날 때 `__exit__()`가 호출되어 정리합니다. **중간에 예외가 나도 `__exit__`은 반드시 실행**되므로, 다음 `try/finally`와 사실상 같습니다.

```python
# with 문:
with open(path) as f:
    data = f.read()
# 위는 아래와 동등:
f = open(path)
try:
    data = f.read()
finally:
    f.close()   # 예외가 나도 반드시 닫힘
```

이 저장소의 실제 용례:

```python
with open(model_path, 'r') as f:   # run.py:37 — 읽기용 파일, 블록 종료 시 자동 닫힘
    model = json.load(f)
with open(output_path, 'w') as f:  # train.py:265 — 쓰기용 파일
    json.dump(model, f)
with urllib.request.urlopen(url, timeout=30) as response:  # download_wikipedia.py:59 — 네트워크 연결도 자동 정리
    html = response.read().decode('utf-8')
```

`with`를 쓰면 `f.close()`를 깜빡하거나 예외로 건너뛰는 실수를 원천적으로 막아, 파일 핸들·소켓 누수를 방지합니다.

### `as`
객체에 **별칭(이름)을 붙입니다**. `with ... as`, `except ... as`, `import ... as`에서 쓰입니다.

```python
with open(model_path, 'r') as f:      # run.py:37 — 열린 파일을 f라는 이름으로
except urllib.error.URLError as e:    # download_wikipedia.py:61 — 예외 객체를 e로
```

---

## 9. 모듈 임포트 (Imports)

### `import`
**다른 모듈(라이브러리)을 불러옵니다**. microgpt는 표준 라이브러리 `math`, `os` 등만 사용합니다.

```python
import os   # gpt.py:9 — 파일 존재 확인 등 운영체제 기능 사용
```

### `from`
모듈에서 **특정 요소만 골라 불러옵니다**(`from 모듈 import 이름`). `import 모듈`과 달리, 가져온 이름을 **모듈 접두어 없이 바로** 쓸 수 있습니다.

```python
import pathlib
pathlib.Path("x")   # import 모듈 → 매번 "모듈." 접두어 필요

from pathlib import Path
Path("x")           # from ... import → 이름만으로 바로 사용
```

이 저장소의 실제 용례:

```python
from pathlib import Path      # download_wikipedia.py:21 — pathlib에서 Path 클래스만
from datetime import datetime # download_wikipedia.py:22 — datetime 모듈에서 datetime 클래스만
```

**여러 개 가져오기 / 별칭:**
```python
from os.path import join, exists, dirname   # 쉼표로 여러 이름을 한 번에
from datetime import datetime as dt         # as로 별칭을 붙일 수도 있음
```

> **주의:** `from 모듈 import *`는 모듈의 모든 공개 이름을 한꺼번에 현재 이름공간에 쏟아붓습니다. 어떤 이름이 어디서 왔는지 알기 어려워지고 기존 이름과 충돌할 수 있어, 일반적으로 **권장되지 않습니다**. 이 저장소에서도 쓰지 않습니다.

---

## 10. 키워드가 아닌 주요 문법 (Non-keyword Syntax)

키워드는 아니지만 이 저장소 코드에 자주 등장하는 핵심 문법·기호들입니다.

### 대괄호 `[]` — 리스트 / 인덱싱 / 슬라이싱
가장 다재다능한 기호로, 세 가지 역할을 합니다.

```python
chars = ['<BOS>'] + sorted(set(''.join(docs)))  # gpt.py:26 — 리스트 리터럴(값들의 나열)
tok_emb = state_dict['wte'][token_id]           # gpt.py:152 — 인덱싱(딕셔너리 키 / 리스트 위치)
q_h = q[hs:hs+head_dim]                          # gpt.py:169 — 슬라이싱(부분 잘라내기)
```

- **리스트 리터럴**: `[a, b, c]`처럼 순서 있는 값들의 모음.
- **인덱싱**: `x[i]`로 i번째 요소(또는 딕셔너리의 키)에 접근.
- **슬라이싱**: `x[start:stop]`로 일부 구간을 잘라 새 리스트를 얻음(`sys.argv[1:]`은 첫 인자를 뺀 나머지).

### 중괄호 `{}` — 딕셔너리 / 집합
```python
state_dict = {'wte': matrix(...), 'wpe': matrix(...)}  # gpt.py:124 — 딕셔너리(키:값)
visited = set()                                        # gpt.py:95 — 빈 집합(중복 없는 모음)
```
- `{'키': 값, ...}`은 **딕셔너리**, `{1, 2, 3}`은 **집합(set)**입니다(빈 집합만은 `set()`으로 만듦 — `{}`는 빈 딕셔너리).

### 소괄호 `()` — 튜플 / 그룹화 / 함수 호출
```python
out = Value(self.data + other.data, (self, other), '+')  # gpt.py:47 — (self, other)는 튜플
def __init__(self, data, _children=(), _op=''):          # gpt.py:38 — ()는 빈 튜플 기본값
scale = (ms + 1e-5) ** -0.5                               # gpt.py:148 — 연산 우선순위 그룹화
```
- **튜플**: `(a, b)`처럼 **변경 불가능한** 순서 모음.
- **함수 호출**: `f(x)`의 괄호. `그룹화`: 수식의 계산 순서를 묶음.

### 컴프리헨션 (Comprehension)
`[]`/`{}` 안에 `for`를 넣어 **반복으로 리스트·딕셔너리를 한 줄에 생성**합니다. 이 코드베이스의 특징적인 문법입니다.

```python
docs = [l.strip() for l in open('input.txt')... if l.strip()]  # gpt.py:21 — 리스트 컴프리헨션(+조건)
stoi = { ch:i for i, ch in enumerate(chars) }                  # gpt.py:28 — 딕셔너리 컴프리헨션
params = [p for mat in state_dict.values() for row in mat for p in row]  # gpt.py:132 — 중첩(평탄화)
```

`for ... if ...`로 필터링까지 할 수 있어, 명시적 반복문보다 간결합니다.

### 삼중 따옴표 `"""..."""` — 여러 줄 문자열 / 독스트링
```python
"""
The most atomic way to train and inference a GPT...
"""                                              # gpt.py:1 — 파일 최상단 독스트링
class Value:
    """Stores a single scalar value and its gradient."""  # gpt.py:36 — 클래스 독스트링
```
- 줄바꿈을 포함한 **여러 줄 문자열**을 만듭니다.
- 함수·클래스·모듈의 **맨 처음**에 두면 **독스트링(docstring)**이 되어, 그 대상의 공식 설명으로 `help()` 등에 노출됩니다.

### f-문자열 `f"..."` — 서식 있는 문자열
```python
print(f"num docs: {len(docs)}")            # gpt.py:23 — 중괄호 안 표현식을 값으로 삽입
state_dict[f'layer{i}.attn_wq'] = matrix(...)  # gpt.py:126 — 동적으로 키 이름 생성
print(f"step {step+1:4d} | loss {loss.data:.4f}")  # gpt.py:227 — :4d, :.4f 서식 지정
```
- `f"..."` 안의 `{표현식}`이 **그 자리에서 값으로 치환**됩니다.
- `{값:.4f}`처럼 콜론 뒤에 **서식 지정자**(소수점 자리수, 폭 등)를 붙일 수 있습니다.

### 언더스코어 `_` — 버리는 값
```python
matrix = lambda nout, nin, std=0.02: [[Value(...) for _ in range(nin)] for _ in range(nout)]  # gpt.py:123
```
반복은 하되 **값 자체는 쓰지 않을 때**, 관례적으로 `_`를 이름으로 씁니다("의도적으로 무시함"을 표현).

### 세미콜론 `;` — 한 줄에 여러 문장
```python
num_steps = int(args[i + 1]); i += 2   # train.py:26 — 두 문장을 한 줄에
```
보통은 한 줄에 한 문장이지만, `;`로 짧은 문장을 이어 붙일 수 있습니다(간결한 인자 파싱 등에 제한적으로 사용).

### 주요 연산자 (`+=`, `//`, `**`, `%`)
```python
self.grad += out.grad          # gpt.py:49 — 복합 대입(기존 값에 더해 저장)
head_dim = n_embd // n_head    # gpt.py:122 — 정수 나눗셈(몫)
scale = (ms + 1e-5) ** -0.5    # gpt.py:148 — 거듭제곱(제곱근 = ** 0.5)
doc = docs[step % len(docs)]   # gpt.py:199 — 나머지(순환 인덱스)
```
| 연산자 | 이름 | 역할 |
|---|---|---|
| `+=` `-=` | 복합 대입 | `x = x + y`의 축약 |
| `//` | 정수 나눗셈 | 나눈 몫(소수점 버림) |
| `**` | 거듭제곱 | `2 ** 3 == 8`, `x ** 0.5`는 제곱근 |
| `%` | 나머지(모듈로) | 순환 인덱스 등에 유용 |

---

## 11. 주요 내장 함수 (Built-in Functions)

키워드는 아니지만 `import` 없이 바로 쓸 수 있는 **내장 함수**들입니다. 이 저장소 코드에서 실제로 호출되는 것들만 목적별로 정리했습니다.

### 시퀀스 생성·순회
```python
for li in range(n_layer):                     # gpt.py:125 — range(n): 0..n-1 정수 시퀀스
for i, ch in enumerate(chars):                # gpt.py:28 — enumerate: (인덱스, 값) 쌍으로 순회
for wi, xi in zip(wo, x):                      # gpt.py:138 — zip: 여러 시퀀스를 짝지어 동시 순회
for v in reversed(topo):                       # gpt.py:105 — reversed: 역순으로 순회(역전파에 사용)
for chunk in iter(lambda: f.read(8192), b''):  # download_wikipedia.py:177 — iter(호출가능, 종료값)
```

| 함수 | 역할 |
|---|---|
| `range(n)` / `range(a, b)` | 정수 시퀀스 생성 (반복 횟수·인덱스) |
| `enumerate(seq)` | 각 요소를 `(순번, 값)` 쌍으로 |
| `zip(a, b, ...)` | 여러 시퀀스를 대응 요소끼리 묶음 |
| `reversed(seq)` | 순서를 뒤집어 순회 |
| `iter(호출가능, 종료값)` | 종료값이 나올 때까지 함수를 반복 호출 |

### 정렬·집합·자료형
```python
chars = ['<BOS>'] + sorted(set(''.join(docs)))  # gpt.py:26 — set으로 중복 제거 후 sorted로 정렬
self._prev = set(_children)                      # gpt.py:42 — 중복 없는 집합 생성
state_dict = {k: [list(row) for row in mat] ...} # run.py:57 — list(): 다른 시퀀스를 리스트로 변환
```

- **`sorted(iterable)`**: 요소를 **정렬한 새 리스트**를 반환합니다. 원본은 그대로 둡니다(리스트의 `.sort()`는 원본을 바꾸는 것과 대비). `sorted(set(...))`는 "**중복 제거 → 정렬**"의 관용구로, 여기서는 문자들을 알파벳 순 어휘로 만듭니다.
- **`set(iterable)`**: **중복 없는 집합**을 만듭니다. `set(''.join(docs))`는 모든 문자에서 고유 문자만 추립니다. 빈 집합은 `set()`으로 만듭니다.
- **`list(iterable)`**: 다른 반복 가능한 객체(튜플·제너레이터 등)를 **리스트로 변환**합니다.

### 집계 (Aggregation)
```python
print(f"num docs: {len(docs)}")               # gpt.py:23 — len: 길이(요소 개수)
return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]  # gpt.py:138 — sum: 총합
max_val = max(val.data for val in logits)     # gpt.py:141 — max: 최댓값(softmax 안정화)
n = min(block_size, len(tokens) - 1)          # gpt.py:201 — min: 최솟값
```

| 함수 | 역할 |
|---|---|
| `len(x)` | 요소 개수(길이) |
| `sum(iterable)` | 모든 요소의 합 (∑에 대응 — [`math.md`](math.md) 참고) |
| `max(iterable)` / `min(iterable)` | 최댓값 / 최솟값 |

### 자료형 변환·검사
```python
num_steps = int(args[i + 1])       # train.py:26 — 문자열 → 정수
temperature = float(args[i + 1])   # run.py:28 — 문자열 → 실수
other = other if isinstance(other, Value) else Value(other)  # gpt.py:46 — 타입 검사
```

- **`int(x)` / `float(x)`**: 문자열이나 수를 **정수 / 실수로 변환**합니다. CLI 인자(항상 문자열)를 숫자로 바꿀 때 필수입니다.
- **`isinstance(객체, 타입)`**: 객체가 그 **타입인지 검사**해 참/거짓을 반환합니다. `Value` 연산에서 상대가 이미 `Value`인지 확인해, 아니면 감싸 주는 데 씁니다.

### 입출력
```python
docs = [... for l in open('input.txt').read()...]  # gpt.py:21 — open: 파일 열기
print(f"vocab size: {vocab_size}")                 # gpt.py:31 — print: 표준 출력
```

- **`open(경로, 모드)`**: 파일을 열어 파일 객체를 반환합니다. 보통 [`with`](#with)와 함께 써서 자동으로 닫습니다.
- **`print(...)`**: 값을 표준 출력에 씁니다. `end=""`(줄바꿈 억제), `flush=True`(즉시 출력) 같은 옵션도 지원합니다.

---

## 참고

이 키워드와 문법들은 이 저장소의 순수 Python 구현(`gpt.py` 243줄 등)에 그대로 쓰여 있습니다. GPT의 알고리즘이 어떤 언어 기능 위에서 구현되는지 확인하려면, 각 예시의 파일·줄 번호를 직접 열어 보세요. 관련 개념은 [`llm.md`](llm.md)(LLM 용어)와 [`math.md`](math.md)(수학 용어)에도 정리되어 있습니다.
