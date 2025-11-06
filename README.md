# System Resource Monitoring System

실시간 시스템 리소스를 모니터링하고 시각화된 PDF 리포트를 생성하는 Python 애플리케이션입니다.

## Features

- **실시간 모니터링**: 시스템 리소스를 실시간으로 추적
- **다양한 메트릭 지원**:
  - CPU 사용률 및 주파수
  - CPU 온도 (지원되는 경우)
  - 메모리 (RAM) 사용률
  - Swap 메모리 사용률
  - 디스크 사용률 및 I/O 속도
  - 네트워크 트래픽 (업로드/다운로드)
  - GPU 사용률 및 온도 (NVIDIA GPU가 있는 경우)
- **시각화**: Matplotlib를 사용한 직관적인 그래프
- **PDF 리포트**: 통계 테이블과 그래프가 포함된 전문적인 PDF 리포트

## Requirements

- Python 3.7+
- Linux 운영체제 (온도 센서 지원)

## Installation

1. 저장소 클론:
```bash
git clone https://github.com/solideoai/solideo_18_1_test.git
cd solideo_18_1_test
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## Usage

기본 실행 (5분간 모니터링, 2초 간격):
```bash
python system_monitor.py
```

실행하면 다음과 같은 작업이 수행됩니다:
1. 5분 동안 2초 간격으로 시스템 리소스 데이터 수집
2. 수집된 데이터로 시각화 그래프 생성
3. 통계 요약과 그래프가 포함된 PDF 리포트 생성

## Output

실행이 완료되면 `system_monitor_report.pdf` 파일이 생성됩니다. 이 파일에는 다음이 포함됩니다:

- **시스템 정보**: CPU 코어 수, 총 메모리, 디스크 용량 등
- **통계 요약**: 각 메트릭의 평균, 최소, 최대, 표준편차
- **시각화 그래프**:
  - CPU 사용률 및 주파수 그래프
  - CPU 온도 그래프
  - 메모리 사용률 그래프
  - 디스크 I/O 속도 그래프
  - 네트워크 트래픽 그래프
  - GPU 정보 그래프 (사용 가능한 경우)

## Customization

모니터링 시간 및 간격을 변경하려면 `system_monitor.py`의 `main()` 함수를 수정하세요:

```python
# 10분간 5초 간격으로 모니터링
monitor = SystemMonitor(duration_minutes=10, interval_seconds=5)
```

## Dependencies

- `psutil`: 시스템 정보 수집
- `matplotlib`: 그래프 생성
- `reportlab`: PDF 리포트 생성
- `numpy`: 통계 계산
- `pillow`: 이미지 처리

## License

MIT License - 자세한 내용은 LICENSE.md 파일을 참조하세요.

## Notes

- CPU 온도는 시스템의 센서 지원 여부에 따라 사용 가능 여부가 결정됩니다.
- GPU 모니터링은 NVIDIA GPU와 nvidia-smi가 설치된 경우에만 작동합니다.
- 일부 메트릭은 관리자 권한이 필요할 수 있습니다.
