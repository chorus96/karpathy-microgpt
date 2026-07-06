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
**컨텍스트 매니저**를 사용해, 블록이 끝나면 자원을 자동으로 정리합니다(예: 파일을 자동으로 닫음).

```python
with open(model_path, 'r') as f:   # run.py:37 — 파일을 열고, 블록 종료 시 자동으로 닫음
```

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
모듈에서 **특정 요소만 골라 불러옵니다**(`from 모듈 import 이름`).

```python
from pathlib import Path   # download_wikipedia.py:21 — pathlib 모듈에서 Path 클래스만 가져옴
```

---

## 참고

이 키워드들은 이 저장소의 순수 Python 구현(`gpt.py` 243줄 등)에 그대로 쓰여 있습니다. GPT의 알고리즘이 어떤 언어 기능 위에서 구현되는지 확인하려면, 각 예시의 파일·줄 번호를 직접 열어 보세요. 관련 개념은 [`llm.md`](llm.md)(LLM 용어)와 [`math.md`](math.md)(수학 용어)에도 정리되어 있습니다.
