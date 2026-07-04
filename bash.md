# Bash 키워드 및 명령 정리

이 문서는 이 저장소의 `*.sh` 파일들(`train.sh`, `persistence/train.sh`, `persistence/run.sh`)에서 **실제로 사용된 Bash 요소**들 — 예약어, 빌트인, 외부 명령, 연산자, 특수 변수 — 을 상세히 정리한 것입니다. 각 항목마다 역할 설명과 함께 저장소 스크립트에서 그대로 가져온 실제 사용 예를 담았습니다.

> 세 스크립트는 구조가 거의 같습니다: **① uv가 없으면 설치하고 ② Python 스크립트를 실행**합니다. 아래 요소들이 이 흐름을 이룹니다.

---

## 1. 셔뱅 (Shebang)

### `#!/usr/bin/env bash`
파일 첫 줄에 오는 **셔뱅(shebang)**. 이 스크립트를 어떤 인터프리터로 실행할지 운영체제에 알려 줍니다. `/usr/bin/env bash`는 `PATH`에서 `bash`를 찾아 실행하므로, bash 설치 위치가 시스템마다 달라도 동작합니다.

```bash
#!/usr/bin/env bash   # train.sh:1 — 모든 스크립트의 첫 줄
```

---

## 2. 엄격 모드: `set` 빌트인 (Strict Mode)

### `set -euo pipefail`
셸의 동작 옵션을 켜는 빌트인 명령. 세 스크립트 모두 두 번째 줄에서 **엄격 모드**를 켜서, 오류를 조용히 넘기지 않고 즉시 중단시킵니다.

```bash
set -euo pipefail   # train.sh:2
```

각 옵션의 의미:

| 옵션 | 이름 | 역할 |
|---|---|---|
| `-e` | errexit | 명령이 하나라도 **실패(0이 아닌 종료 코드)하면 즉시 스크립트 종료** |
| `-u` | nounset | **정의되지 않은 변수를 사용하면 오류** 처리 (오타로 인한 버그 방지) |
| `-o pipefail` | pipefail | 파이프(`|`)에서 **하나의 명령이라도 실패하면 전체를 실패로** 간주 |

이 셋을 합친 `-euo pipefail`은 견고한 스크립트를 위한 관용적 표준 설정입니다.

---

## 3. 조건문 (Conditionals)

### `if` / `then` / `fi`
**조건 분기 예약어**. `if 조건; then ... fi` 형태로, 조건이 참(종료 코드 0)일 때 블록을 실행합니다. `fi`는 `if`를 거꾸로 쓴 것으로 블록의 끝을 나타냅니다.

```bash
if ! command -v uv &>/dev/null; then   # train.sh:5
    echo "uv not found, installing..."
    ...
fi                                     # train.sh:18
```

### `!` (부정 / NOT)
뒤에 오는 명령의 **종료 코드를 뒤집습니다**(성공↔실패). 위 예에서 `! command -v uv`는 "uv를 찾지 **못하면** 참"이 되어, uv가 없을 때만 설치 블록을 실행합니다.

---

## 4. 빌트인 명령 (Builtins)

### `command`
셸 빌트인. `command -v uv`는 **`uv`가 실행 가능한지(설치되어 있는지) 확인**하고 그 경로를 출력합니다. 여기서는 존재 여부만 필요하므로 출력은 `/dev/null`로 버립니다.

```bash
if ! command -v uv &>/dev/null; then   # train.sh:5 — uv 설치 여부 확인
```

### `echo`
인자로 받은 **문자열을 표준 출력에 출력**합니다. 진행 상황·오류 메시지 표시에 쓰입니다.

```bash
echo "uv not found, installing..."   # train.sh:6
echo "ERROR: uv installation failed" >&2   # train.sh:14 — 오류는 표준 에러로
```

### `export`
변수를 **환경 변수로 내보내** 하위 프로세스에서도 보이게 합니다. 여기서는 uv 설치 경로를 `PATH` 앞에 추가합니다.

```bash
export PATH="$HOME/.local/bin:$PATH"   # train.sh:10 — uv를 현재 셸에서 즉시 사용 가능하게
```

### `exit`
스크립트를 **지정한 종료 코드로 즉시 종료**합니다. `exit 1`은 "실패"를 뜻하는 관례적 코드입니다.

```bash
exit 1   # train.sh:15 — uv 설치 실패 시 오류로 종료
```

### `cd` / `pwd`
- `cd`: **디렉터리 이동** 빌트인.
- `pwd`: **현재 작업 디렉터리 경로 출력**(print working directory) 빌트인.

`persistence/` 스크립트들은 이 둘을 조합해 스크립트 자신의 절대 경로를 구합니다(아래 6절 `BASH_SOURCE` 참고).

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # persistence/train.sh:4
```

---

## 5. 외부 명령 (External Commands)

### `curl`
URL에서 데이터를 내려받는 도구. uv 설치 스크립트를 다운로드하는 데 쓰입니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # train.sh:7
```
자주 쓰는 플래그: `-L`(리디렉션 따라가기), `-s`(진행 표시 숨김/조용히), `-S`(조용해도 오류는 표시), `-f`(HTTP 오류 시 실패 처리).

### `sh`
POSIX 셸. 위에서 `curl`이 내려받은 설치 스크립트를 **파이프로 받아 실행**합니다.

### `dirname`
경로에서 **디렉터리 부분만 추출**합니다(예: `/a/b/c.sh` → `/a/b`). 스크립트 위치 계산에 쓰입니다.

```bash
$(dirname "${BASH_SOURCE[0]}")   # persistence/run.sh:4 — 스크립트가 있는 폴더 경로
```

### `env`
셔뱅에서 `PATH`를 검색해 `bash`를 찾아 실행합니다(1절 참고).

### `uv`
Python 패키지·실행 관리 도구. `uv run`은 필요한 의존성 환경에서 Python 스크립트를 실행합니다.

```bash
uv run gpt.py                        # train.sh:20
uv run "$SCRIPT_DIR/train.py" "$@"   # persistence/train.sh:20 — 추가 인자를 그대로 전달
```

---

## 6. 변수와 확장 (Variables & Expansion)

### 변수 참조 `$변수` / `"${변수}"`
`$이름` 또는 `${이름}`으로 변수 값을 꺼냅니다. 공백 문제를 막기 위해 보통 큰따옴표로 감쌉니다.

### `$HOME`
사용자의 **홈 디렉터리 경로**를 담은 환경 변수(예: `/home/user`).

```bash
export PATH="$HOME/.local/bin:$PATH"   # train.sh:10
```

### `$PATH`
셸이 **명령 실행 파일을 찾는 디렉터리 목록**(콜론으로 구분). 앞에 경로를 추가하면 그 위치의 명령이 우선 검색됩니다.

### `${BASH_SOURCE[0]}`
현재 실행 중인 **스크립트 파일 자신의 경로**를 담은 배열. `[0]`은 첫 번째 요소(현재 파일)입니다. 어디서 호출하든 스크립트 위치를 정확히 알아내기 위해 씁니다.

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # persistence/train.sh:4
```

### `"$@"`
스크립트에 전달된 **모든 인자를 각각 따로** 넘겨주는 특수 변수. `persistence` 스크립트는 이를 이용해 `--steps`, `--samples` 같은 옵션을 Python 스크립트로 그대로 전달합니다.

```bash
uv run "$SCRIPT_DIR/run.py" "$@"   # persistence/run.sh:20
```

### 명령 치환 `$(...)`
괄호 안 **명령의 출력 결과를 그 자리에 삽입**합니다. 중첩도 가능합니다.

```bash
echo "uv installed successfully: $(uv --version)"   # train.sh:16 — 버전 문자열을 삽입
SCRIPT_DIR="$(cd ... && pwd)"                        # 명령 결과를 변수에 저장
```

---

## 7. 연산자와 리디렉션 (Operators & Redirection)

### 파이프 `|`
앞 명령의 **표준 출력을 뒤 명령의 표준 입력으로** 연결합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # train.sh:7 — 내려받아 바로 실행
```

### AND 리스트 `&&`
앞 명령이 **성공(종료 코드 0)했을 때만** 뒤 명령을 실행합니다.

```bash
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # cd가 성공해야 pwd 실행
```

### `&>/dev/null`
명령의 **표준 출력과 표준 에러를 모두** `/dev/null`(빈 공간, "블랙홀")로 보내 **완전히 버립니다**. 존재 여부만 확인할 때 출력 소음을 없애는 용도입니다.

```bash
command -v uv &>/dev/null   # train.sh:5 — 결과 출력은 버리고 종료 코드만 사용
```

### `>&2`
명령의 출력을 **표준 에러(stderr, 파일 디스크립터 2)로** 보냅니다. 오류 메시지는 일반 출력과 구분하기 위해 stderr로 내보내는 것이 관례입니다.

```bash
echo "ERROR: uv installation failed" >&2   # train.sh:14
```

### 종료 코드 (Exit Code)
모든 명령은 끝나면 0~255의 **종료 코드**를 남깁니다(0=성공, 그 외=실패). `if`, `!`, `&&`, `set -e`, `pipefail`이 모두 이 종료 코드를 기준으로 동작합니다.

---

## 참고

이 요소들은 이 저장소의 세 실행 스크립트에 그대로 쓰여 있습니다. `train.sh`는 가장 단순한 형태이고, `persistence/train.sh`·`persistence/run.sh`는 스크립트 위치 계산(`BASH_SOURCE`)과 인자 전달(`"$@"`)이 추가된 형태입니다. 관련 문서: [`python.md`](python.md)(Python 키워드), [`llm.md`](llm.md)(LLM 용어), [`math.md`](math.md)(수학 용어).
