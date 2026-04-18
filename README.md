# JWB Semiconductor Anomaly Detection

이 프로젝트는 반도체 생산 장비의 센서 데이터를 기반으로 이상 감지를 수행하는 Autoencoder 모델을 구현한 것입니다. 특히 RF Generator와 Matching Network의 고장 진단, 분석, 수리 업무에 적용할 수 있습니다.

## 목차

- [소개](#소개)
- [주요 기능](#주요-기능)
- [설치 방법](#설치-방법)
- [사용법](#사용법)
- [API 엔드포인트](#api-엔드포인트)
- [모델 학습](#모델-학습)
- [테스트](#테스트)
- [프로젝트 구조](#프로젝트-구조)
- [기여](#기여)
- [라이선스](#라이선스)

## 소개

이 프로젝트는 PyTorch 기반 Autoencoder를 사용하여 센서 데이터를 학습하고, 재구성 오차를 통해 이상을 감지합니다. 초기에는 반도체 센서(temp, pressure, vibration, voltage)를 대상으로 했으나, RF 장비에 특화된 버전으로 확장되었습니다.

RF 장비의 주요 센서:
- Forward Power (W): 들어가는 전력
- Reflected Power (W): 반사 전력 (핵심 지표)
- Frequency (MHz): 주파수 (보통 13.56)
- Impedance (Ω): 매칭 상태
- Voltage / Current: 전기 상태
- Temperature (°C): 과열 여부

## 주요 기능

- **실시간 이상 감지**: 센서 데이터를 입력받아 이상 여부를 판별
- **Autoencoder 모델**: 비지도 학습으로 정상 데이터를 학습
- **FastAPI 기반 API**: RESTful API로 쉽게 통합 가능
- **확장성**: 다양한 센서 데이터에 적용 가능
- **테스트 커버리지**: 유닛 테스트 포함

## 설치 방법

1. **Python 환경 설정**:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   ```

2. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

3. **모델 및 데이터 준비**:
   - 기본 모델: `models/semiconductor_autoencoder.pth`
   - RF 모델: `models/rf_semiconductor_autoencoder.pth` (학습 필요)

## 사용법

### 기본 실행
```bash
python app.py
```

### API 서버 실행
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 테스트 실행
```bash
pytest tests/
```

## API 엔드포인트

### GET /health
모델 초기화 상태 확인
```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.028
}
```

### POST /infer
센서 데이터로 이상 감지
```json
// 요청 (반도체 버전)
{
  "temp": 30.0,
  "pressure": 1.0,
  "vibration": 0.01,
  "voltage": 2.0
}

// 응답
{
  "reconstruction_score": 0.0025,
  "threshold": 0.028,
  "is_anomaly": false
}
```

RF 버전의 경우 센서 필드가 다름 (forward_power, reflected_power 등).

## 모델 학습

### 기본 모델 학습
```bash
python semiconductor_autoencoder.py  # 학습 코드 실행 (별도 구현 필요)
```

### RF 모델 학습
1. 데이터 생성:
   ```bash
   python generate_rf_data.py
   ```

2. 모델 학습:
   ```bash
   python train_rf_model.py
   ```

3. 평가:
   ```bash
   python eval_rf_anomaly.py
   ```

평가 결과 (RF 모델):
- AUC: 0.997
- 정확도: 99.3%
- F1 Score: 0.97

## 테스트

테스트는 `tests/test_app.py`에 구현되어 있습니다.
```bash
pytest tests/test_app.py
```

테스트 커버리지:
- 헬스 체크
- 정상 입력 이상 감지
- 잘못된 입력 검증

## 프로젝트 구조

```
jbw_project/
├── app.py                          # FastAPI 메인 앱
├── semiconductor_autoencoder.py    # Autoencoder 모델 및 학습 코드
├── generate_rf_data.py             # RF 데이터 생성 스크립트
├── train_rf_model.py               # RF 모델 학습 스크립트
├── eval_rf_anomaly.py              # RF 모델 평가 스크립트
├── eval_anomaly.py                 # 기본 모델 평가
├── requirements.txt                # 의존성
├── README.md                       # 이 파일
├── sensor_normal_1000.csv          # 기본 정상 데이터
├── sensor_normal_with_anomaly_1000.csv  # 기본 이상 데이터
├── rf_sensor_normal_1000.csv       # RF 정상 데이터
├── rf_sensor_anomaly_1000.csv      # RF 이상 데이터
├── models/
│   ├── semiconductor_autoencoder.pth    # 기본 모델
│   └── rf_semiconductor_autoencoder.pth  # RF 모델
└── tests/
    └── test_app.py                 # 유닛 테스트
```

## 기여

기여를 환영합니다! 이슈나 PR을 통해 참여해주세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.