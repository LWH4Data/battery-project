# 배터리 데이터 안내

원본 출처는 [Kaggle — Data-driven prediction of battery cycle life](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)다. 원본 MAT 파일은 `.gitignore`의 `data/raw/`에 따라 Git 추적에서 제외한다. 저장소를 받은 뒤 원본을 별도로 확보한다.

## 배치별 경로와 역할

프로젝트 루트를 기준으로 다음 경로에 파일을 둔다.

```text
data/raw/
├── batch1/2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── batch2/2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── batch3/2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

| 배치 | 원본 셀 | 유효 Cycle Life | 역할 |
| --- | --- | --- | --- |
| Batch 1 | 46 | 46 | Day 1 EDA, Day 2 학습·CV·Hold-out |
| Batch 2 | 47 | 39 | Day 1 EDA, Day 2 필수 최종 Test |
| Batch 3 | 46 | 44 | Day 1 EDA, Day 2 선택 추가 Test |

원본 파일의 SHA-256·크기·수정 시각은 [configs/data-snapshot.json](../configs/data-snapshot.json)에 기록했다. 사용 파일이 같은지 확인하고, 다른 파일을 동일한 스냅샷으로 취급하지 않는다. 원본 139셀과 수명 결측 10셀을 보존했다.

## 데이터 준비 범위

- 원본 확인: `notebooks/00_data_check.ipynb`.
- Day 1: 모든 셀의 사이클 summary와 실제 cycle 10·100의 `Vdlin/Qdlin`을 준비한다. 초기 상세 전류는 필요한 셀·사이클에서 직접 읽는다.
- 모델 입력 후보: `outputs/day1/early_feature_candidates.csv`, 셀당 한 행. v01과 최종 v02는 `delta_q_log10var`만 사용한다.
- `interim/`: 반복 읽기 비용을 줄일 때 사용하는 중간 저장소. 현재 전체 상세값을 별도 파일로 저장하지 않았다.
- `processed/`: 향후 별도 모델 입력을 저장할 위치. 이번 단일 피처 입력은 기존 Day 1 CSV를 그대로 사용한다.

설치·실행 순서와 검증 범위는 [프로젝트 README](../README.md)를 참고한다. 셀·사이클 제외, 결측 대체, 보간·클리핑은 임의로 적용하지 않는다.

## 원본 없이 최종 결과 확인

공개 저장소의 피처 CSV·분할·v02 지표/예측으로 [03_day2_final_model.ipynb](../notebooks/03_day2_final_model.ipynb)를 실행할 수 있다. 기존 결과가 있으면 열람만 하며 MAT를 다시 읽거나 Test를 다시 예측하지 않는다. 원본 MAT는 00·01 원본 확인과 EDA를 재실행할 때 필요하다. 01은 원본 크기·SHA-256을 검증한다. Batch 2는 47셀 예측을 보존하고 수명 미상 8셀을 지표에서만 제외했다.
