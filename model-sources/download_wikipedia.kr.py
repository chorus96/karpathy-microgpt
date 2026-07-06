#!/usr/bin/env -S uv run
"""
학습용 영어 Wikipedia 데이터베이스 덤프를 내려받는다.

이 스크립트는 dumps.wikimedia.org에서 영어 Wikipedia 덤프를 다운로드한다.
권장 덤프는 pages-articles-multistream.xml.bz2이며 다음을 포함한다:
- 현재 리비전만 (히스토리 없음)
- 토론(talk)·사용자 페이지 없음
- 압축 크기: ~25 GB
- 압축 해제 크기: ~105 GB

참고: https://en.wikipedia.org/wiki/Wikipedia:Database_download

[이 파일은 model-sources/download_wikipedia.py의 한국어 주석 버전입니다.
코드는 원본과 동일하며, 주석만 한국어로 옮기고 이해를 돕는 설명을 덧붙였습니다.]
"""

import os
import sys
import hashlib          # MD5 체크섬 계산
import argparse         # CLI 인자 파싱
import urllib.request   # HTTP 다운로드
import urllib.error     # HTTP/URL 오류 처리
from pathlib import Path
from datetime import datetime


# Wikipedia 덤프 기본 URL
DUMP_BASE_URL = "https://dumps.wikimedia.org/enwiki"

# 사용 가능한 덤프 유형들 (파일명 템플릿의 {date}는 나중에 날짜로 치환)
DUMP_TYPES = {
    "articles": {
        "filename": "enwiki-{date}-pages-articles-multistream.xml.bz2",
        "index": "enwiki-{date}-pages-articles-multistream-index.txt.bz2",
        "description": "Current article revisions only (recommended, ~25GB compressed)",
    },
    "abstracts": {
        "filename": "enwiki-{date}-abstract.xml.gz",
        "index": None,
        "description": "Page abstracts only (~2GB compressed)",
    },
    "titles": {
        "filename": "enwiki-{date}-all-titles-in-ns0.gz",
        "index": None,
        "description": "Article titles only (~100MB compressed)",
    },
    "meta-current": {
        "filename": "enwiki-{date}-pages-meta-current.xml.bz2",
        "index": None,
        "description": "Current revisions, all pages including talk (~35GB compressed)",
    },
}


def get_latest_dump_date() -> str:
    """Wikimedia에서 가장 최근의 덤프 날짜를 가져온다."""
    url = f"{DUMP_BASE_URL}/"
    print(f"Fetching available dumps from {url}...")

    try:
        # 디렉터리 목록 HTML을 받아온다
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"Error fetching dump list: {e}")
        sys.exit(1)

    # 디렉터리 목록에서 날짜를 파싱 (형식: YYYYMMDD)
    import re
    dates = re.findall(r'href="(\d{8})/"', html)

    if not dates:
        print("Error: Could not find any dump dates")
        sys.exit(1)

    # 정렬 후 가장 최신 날짜 선택 (내림차순의 첫 번째)
    dates.sort(reverse=True)
    latest = dates[0]
    print(f"Latest dump date: {latest}")

    return latest


def get_dump_status(date: str) -> dict:
    """특정 덤프의 상태를 확인한다."""
    status_url = f"{DUMP_BASE_URL}/{date}/dumpstatus.json"

    try:
        with urllib.request.urlopen(status_url, timeout=30) as response:
            import json
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError:
        return None  # 상태 확인 실패 시 None


def download_file(url: str, output_path: Path, resume: bool = True) -> bool:
    """
    진행률 표시와 (선택적) 이어받기를 지원하는 파일 다운로드.

    Args:
        url: 다운로드할 URL
        output_path: 저장할 로컬 경로
        resume: 부분 다운로드를 이어받을지 여부

    Returns:
        성공하면 True, 아니면 False
    """
    # 기존 부분 다운로드 확인
    start_byte = 0
    if resume and output_path.exists():
        start_byte = output_path.stat().st_size  # 이미 받은 크기
        print(f"Resuming download from byte {start_byte:,}")

    # 이어받기를 위한 Range 헤더 설정
    request = urllib.request.Request(url)
    if start_byte > 0:
        request.add_header('Range', f'bytes={start_byte}-')  # 이 바이트부터 요청

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            # 전체 파일 크기 얻기
            content_length = response.headers.get('Content-Length')
            if content_length:
                total_size = int(content_length) + start_byte
            else:
                total_size = None

            # 서버가 Range(이어받기)를 지원하는지 확인
            content_range = response.headers.get('Content-Range')
            if start_byte > 0 and not content_range:
                print("Server doesn't support resume, starting from beginning")
                start_byte = 0

            # 적절한 모드로 파일 열기 (이어받기면 append, 아니면 write)
            mode = 'ab' if start_byte > 0 else 'wb'

            downloaded = start_byte
            chunk_size = 1024 * 1024  # 1MB 청크
            last_progress_time = datetime.now()

            with open(output_path, mode) as f:
                while True:
                    chunk = response.read(chunk_size)  # 1MB씩 스트리밍
                    if not chunk:
                        break  # 다 받으면 종료

                    f.write(chunk)
                    downloaded += len(chunk)

                    # 1초마다 진행률 출력
                    now = datetime.now()
                    if (now - last_progress_time).seconds >= 1:
                        if total_size:
                            progress = downloaded / total_size * 100
                            print(f"\rProgress: {downloaded:,} / {total_size:,} bytes ({progress:.1f}%)", end='', flush=True)
                        else:
                            print(f"\rDownloaded: {downloaded:,} bytes", end='', flush=True)
                        last_progress_time = now

            print()  # 진행률 뒤 줄바꿈
            return True

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e}")
        return False
    except KeyboardInterrupt:
        # 사용자가 중단(Ctrl+C)하면 안전하게 종료 (다시 실행하면 이어받기)
        print("\nDownload interrupted. Run again to resume.")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """파일의 MD5 체크섬을 검증한다."""
    print(f"Verifying MD5 checksum for {filepath.name}...")

    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        # 8KB씩 읽어 해시를 갱신 (대용량 파일도 메모리 부담 없이)
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)

    actual_md5 = md5_hash.hexdigest()

    if actual_md5 == expected_md5:
        print(f"MD5 verified: {actual_md5}")
        return True
    else:
        # 체크섬 불일치 → 파일 손상 가능성
        print(f"MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def download_wikipedia(
    output_dir: str = "data",
    dump_type: str = "articles",
    dump_date: str = None,
    include_index: bool = True,
    verify: bool = True,
) -> None:
    """
    영어 Wikipedia 덤프를 내려받는다 (전체 흐름 오케스트레이터).

    Args:
        output_dir: 다운로드 파일을 저장할 디렉터리
        dump_type: 덤프 유형 (articles, abstracts, titles, meta-current)
        dump_date: 특정 덤프 날짜(YYYYMMDD) 또는 None(최신)
        include_index: 인덱스 파일도 받을지 (multistream용)
        verify: 다운로드 후 MD5 검증 여부
    """
    if dump_type not in DUMP_TYPES:
        # 알 수 없는 유형 → 사용 가능한 목록 안내 후 종료
        print(f"Unknown dump type: {dump_type}")
        print(f"Available types: {', '.join(DUMP_TYPES.keys())}")
        sys.exit(1)

    dump_info = DUMP_TYPES[dump_type]
    print(f"Dump type: {dump_type}")
    print(f"Description: {dump_info['description']}")

    # 덤프 날짜 결정 (미지정이면 최신)
    if dump_date is None:
        dump_date = get_latest_dump_date()

    # 덤프 상태 확인
    status = get_dump_status(dump_date)
    if status:
        jobs = status.get('jobs', {})
        articles_status = jobs.get('articlesmultistreamdump', {}).get('status', 'unknown')
        print(f"Dump status: {articles_status}")
        if articles_status != 'done':
            print("Warning: Dump may not be complete yet")  # 아직 생성 중일 수 있음

    # 출력 디렉터리 생성 (없으면 만들고, 있으면 통과)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파일 URL 구성
    filename = dump_info['filename'].format(date=dump_date)  # {date}를 실제 날짜로 치환
    file_url = f"{DUMP_BASE_URL}/{dump_date}/{filename}"
    local_path = output_path / filename

    print(f"\nDownloading: {filename}")
    print(f"URL: {file_url}")
    print(f"Output: {local_path}")

    # 본 파일 다운로드
    success = download_file(file_url, local_path)

    if not success:
        print("Download failed!")
        sys.exit(1)

    print(f"Downloaded: {local_path}")
    print(f"Size: {local_path.stat().st_size:,} bytes")

    # 요청되었고 존재한다면 인덱스 파일도 다운로드
    if include_index and dump_info.get('index'):
        index_filename = dump_info['index'].format(date=dump_date)
        index_url = f"{DUMP_BASE_URL}/{dump_date}/{index_filename}"
        index_path = output_path / index_filename

        print(f"\nDownloading index: {index_filename}")
        download_file(index_url, index_path)

    # 요청되면 체크섬 검증
    if verify:
        md5_url = f"{DUMP_BASE_URL}/{dump_date}/{filename}-md5"
        try:
            with urllib.request.urlopen(md5_url, timeout=30) as response:
                expected_md5 = response.read().decode('utf-8').split()[0]
                verify_md5(local_path, expected_md5)
        except urllib.error.URLError:
            print("Could not fetch MD5 checksum for verification")

    print("\nDownload complete!")
    print(f"\nTo extract the dump, use: bunzip2 -k {local_path}")  # 압축 해제 안내
    print("Note: Extraction will require ~100GB of disk space for the full articles dump")


def main():
    # argparse로 CLI 인터페이스 정의
    parser = argparse.ArgumentParser(
        description="Download English Wikipedia database dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download latest articles dump
  %(prog)s --type abstracts          # Download abstracts only
  %(prog)s --type titles             # Download article titles only
  %(prog)s --date 20260101           # Download specific date
  %(prog)s --output ./wikipedia      # Custom output directory

Reference: https://en.wikipedia.org/wiki/Wikipedia:Database_download
        """
    )

    parser.add_argument(
        '--output', '-o',
        default='data',
        help='Output directory (default: data)'  # 출력 디렉터리
    )

    parser.add_argument(
        '--type', '-t',
        choices=list(DUMP_TYPES.keys()),
        default='articles',
        help='Type of dump to download (default: articles)'  # 덤프 유형
    )

    parser.add_argument(
        '--date', '-d',
        help='Dump date in YYYYMMDD format (default: latest)'  # 특정 날짜
    )

    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Skip downloading the index file'  # 인덱스 건너뛰기
    )

    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip MD5 verification'  # MD5 검증 건너뛰기
    )

    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available dump dates and exit'  # 날짜 목록만 출력
    )

    args = parser.parse_args()

    if args.list_dates:
        # --list-dates: 최신 10개 날짜만 보여주고 종료
        url = f"{DUMP_BASE_URL}/"
        print(f"Fetching available dumps from {url}...")
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode('utf-8')
        import re
        dates = re.findall(r'href="(\d{8})/"', html)
        dates.sort(reverse=True)
        print("\nAvailable dump dates (most recent first):")
        for date in dates[:10]:
            print(f"  {date}")
        return

    # 실제 다운로드 실행 (CLI 인자를 함수 인자로 전달)
    download_wikipedia(
        output_dir=args.output,
        dump_type=args.type,
        dump_date=args.date,
        include_index=not args.no_index,   # --no-index면 False
        verify=not args.no_verify,         # --no-verify면 False
    )


if __name__ == '__main__':
    # 스크립트로 직접 실행될 때만 main() 호출 (import될 때는 실행 안 함)
    main()
