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
- 모델 입력 후보: `outputs/day1/early_feature_candidates.csv`, 셀당 한 행. 현재 v01은 `delta_q_log10var`만 사용한다.
- `interim/`: 반복 읽기 비용을 줄일 때 사용하는 중간 저장소. 현재 전체 상세값을 별도 파일로 저장하지 않았다.
- `processed/`: 최종 피처·처리 규칙이 확정되면 모델 입력 데이터를 저장할 위치.

설치·실행 순서와 검증 범위는 [프로젝트 README](../README.md)를 참고한다. 셀·사이클 제외, 결측 대체, 보간·클리핑은 임의로 적용하지 않는다.
