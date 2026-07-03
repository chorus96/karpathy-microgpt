# 모델 소스 (Model Sources)

학습 데이터를 다운로드하기 위한 도구들입니다.

## Wikipedia 덤프

GPT 학습을 위해 영어 Wikipedia를 다운로드합니다.

```bash
# 도움말 보기
uv run download_wikipedia.py --help

# 전체 문서 다운로드 (압축 시 ~25GB)
uv run download_wikipedia.py

# 제목만 다운로드 (~100MB, 테스트에 적합)
uv run download_wikipedia.py --type titles

# 요약 다운로드 (~2GB)
uv run download_wikipedia.py --type abstracts

# 사용자 지정 출력 디렉터리
uv run download_wikipedia.py --output ../data/wikipedia

# 사용 가능한 덤프 날짜 목록 보기
uv run download_wikipedia.py --list-dates
```

### 덤프 유형

| 유형 | 크기 | 설명 |
|------|------|------|
| `articles` | ~25GB | 전체 문서 텍스트 (기본값) |
| `abstracts` | ~2GB | 페이지 요약 |
| `titles` | ~100MB | 문서 제목만 |
| `meta-current` | ~35GB | 토론(talk)을 포함한 모든 페이지 |

### 참고 사항

- 다운로드가 중단되면 자동으로 재개됩니다
- MD5 체크섬이 기본적으로 검증됩니다
- 전체 문서 덤프는 압축 해제 시 ~100GB로 확장됩니다
