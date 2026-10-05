# Day 2 — v01 첫 선형회귀 기준 모델

- 작성일: 2026-10-02, Asia/Seoul
- 예측 단위: 배터리 셀 1개당 한 행.
- 입력: `delta_q_log10var` 한 개. 첫 10·100사이클의 ΔQ 분산에 log10을 적용한 값.
- 타깃: 원본 `cycle_life`(사이클), 타깃 변환 없음.
- 모델: `StandardScaler → LinearRegression`. 각 CV fold의 train에서만 fit.
- 분할: 원문 `policy_readable` 그룹의 20% Hold-out, 남은 개발 부분에서 그룹 5-fold CV, seed 42.

## 실제 분할

| 구분 | 셀 수 | 충전 조건 그룹 수 | 사용 |
| --- | --- | --- | --- |
| Batch 1 전체 | 46 | 23 | 원본 셀 모두 보존 |
| 개발 부분 | 35 | 18 | CV와 개발 기준 모델 학습 |
| Hold-out | 11 | 5 | 모델 선택 후 평가용으로 보관 |

20%는 그룹 수 기준이다. 올림으로 5개 그룹이 보관되고, 이 그룹들의 셀 수는 11개다.
그룹별 Hold-out·CV 분리와 개발 셀당 1회의 검증 예측을 검사했다.

## 첫 기준 성능

| 구분 | MAPE (%) | 비고 |
| --- | --- | --- |
| Train (Batch 1 CV) | 7.2398 | 5개 validation fold MAPE의 산술평균 |
| Valid (Batch 1 Hold-out) | 미평가 | 후보 선택 후 평가 |
| Test (Batch 2) | 미평가 | 설정 확정 후 최종 평가 |

- Fold 간 표본 표준편차: **2.4718%p**. 신뢰구간을 뜻하지 않는다.
- 보조 fold 평균: MAE **60.14사이클**, RMSE **72.26사이클**.
- 개발 셀 전체의 검증 예측을 합친 pooled OOF MAPE: 7.1934%. 셀 수로 가중되는 값이어서 주 지표인 fold 평균과 구분한다.

| Fold | Train 셀 | Validation 셀 | MAPE (%) |
| --- | --- | --- | --- |
| 1 | 28 | 7 | 10.3862 |
| 2 | 27 | 8 | 5.9497 |
| 3 | 27 | 8 | 6.1454 |
| 4 | 29 | 6 | 4.4729 |
| 5 | 29 | 6 | 9.2449 |

![Fold별 MAPE와 개발 셀의 CV 예측](/Users/lwh/Desktop/battery-project/outputs/day2/v01_baseline_cv.png)

## 해석

ΔQ log분산 하나로 개발 부분의 평균 상대 오차 약 7.24%를 얻었다.
Fold별 오차는 4.47%~10.39%로 달라진다.
이후 피처 추가·규제 회귀가 같은 CV 분할에서 개선되는지 확인할 기준으로 사용한다.
CV에서 좋은 성능을 얻었다는 사실만으로 다른 배치의 성능이나 논문 목표 달성을 주장하지 않는다.
최종 모델 성능은 후보·설정을 고정한 뒤 별도 Hold-out과 Batch 2에서 확인한다.

## 재현성과 다음 결정

- 분할: `artifacts/splits/batch1_group_holdout_cv.json`, `split_assignments.csv`.
- 실험: `configs/experiments/v01_baseline.json`.
- 결과: `artifacts/experiments/v01_baseline/`의 실제 CV 지표·예측·모델·실행 manifest.
- 저장 모델은 개발 35셀에 학습한 기준 모델이며, 최종 선택 모델이 아니다.
- 실행 노트북: `notebooks/02_day2_modeling.ipynb`, 코드 셀 6개 모두 정상 실행.
- 확장 피처, 품질 처리, 규제 강도·Optuna 탐색 조건은 다음 단계에서 사용자와 결정한다.
