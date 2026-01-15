# re-ranking

```bash
├── data
│   ├── input       # 모델의 input으로 들어가는 파일 모음
│   ├── output      # 모델의 output으로 나오는 파일 모음
│   │   ├── logs        # 1) metric
│   │   └── oof         # 2) oof
├── results
│   ├── analysis    # 데이터 분석 결과
│   └── graphs      # 시각화 결과
└── src
    ├── analysis    # 데이터 분석 코드
    ├── graphs      # 시각화 코드
    └── models      # 모델 코드(residual-sign model -> re-ranking model)