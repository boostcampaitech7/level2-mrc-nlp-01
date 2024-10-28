# NaverBoostCamp AI Tech 7기: MRC(RAG) 프로젝트

## 1. 프로젝트 개요

### 주제: Open-Domain Question Answering

본 프로젝트는 사전에 구축된 Wikipedia 데이터셋에서 질문에 대답할 수 있는 알맞은 문서를 Retriever를 활용하여 추출하고, Reader를 통해 추출한 문서에서 정답을 찾거나 생성하는 Retriever-Reader two-stage 모델을 구현하는 것을 목표로 한다.

### 평가 지표

- **주요 지표**: EM (Exact Match)
- **참고 지표**: F1 Score

### 최종 순위
| 분류                | 순위 | EM   | F1   |
|---------------------|------|------|------|
| private (최종 순위) | 5    | 67.22 | 77.88 |
| public (중간 순위)  | 12   | 64.17 | 74.89 |

### 데이터 구성

| 분류 | 샘플 수 | 용도 | 공개여부 |
|------|---------|------|----------|
| Wikipedia | 57k | Retriever 학습용 | 모든 정보 공개 (text, corpus_source, url, domain, title, author, html, document_id) |
| Train | 3952 | Reader 학습용 | 모든 정보 공개(id, question, context, answers, document_id, title) |
| Validation | 240 | Reader 학습용 | 모든 정보 공개(id, question, context, answers, document_id, title) |
| Test (Public) | 240 | 제출용 | id, question만 공개 |
| Test (Private) | 360 | 제출용 | id, question만 공개 |

### 개발 환경

- **서버**: AI Stages GPU (Tesla V100-SXM2) * 4EA
- **기술 스택**: Python, Transformers, PyTorch, Pandas, WandB, Hugging Face, Matplotlib
- **운영체제**: Linux
- **협업 도구**: GitHub, Notion, Slack, Zoom

## 2. 팀 구성 및 역할

| 팀원 | 역할 |
|------|------|
| 박준성 | **협업 관리** (이슈/PR 템플릿 추가), **코드 리팩토링** (베이스라인 코드 재작성, Config 클래스), **EDA 서버 개설 및 관리** (Page 시스템, 훈련용 데이터셋 EDA 페이지), **Cross Encoder 구현**, **Retrieval 성능 측정** EDA (TF-IDF, BM25), **앙상블 구현** (soft voting), **모델 파이프라인 병합** (Retriever-Reader 연결), **데이터 증강** (KorQuad) |
| 이재백 | **Dense Retriever 구현**, **Negative Sampling 구현** (in-batch, random, Hard negative sampling), **모델 조사 및 실험** |
| 강신욱 | **모델 탐색 및 실험**, **기능 향상 시도** (가중치를 활용한 sparse와 dense 리트리버 결합, GPT를 활용한 dense retrieval의 네거티브 샘플링 생성) |
| 홍성균 | **Sparse Retrieval 구현 (BM25)**, **코드 리팩토링** (main.py 병합, CPU 병렬처리, Reader/Retrieval Tokenizer 분리 적용), **EDA 서버 증축** (DataEDA, TokenizerEDA), **실험 편의성 개선** (WandB 적용), **Bug Fix** (하이퍼파라미터 미적용 수정, Argumentparser 미적용 수정), **데이터 증강** (KorQuAD와의 병합) |
| 백승우 | **Generation-based Reader 모델 구현**, **PEFT** (Parameter Efficient Fine-Tuning) **기법 중** **LoRA** (Low-Rank Adaptation) **구현**, **Generation-based 모델 조사 및 실험** |
| 김정석 | **한자 등 각 유니코드 블록에 해당하는 문자를 제거하는 전처리 함수 작성 및 EDA** |

## 3. 프로젝트 파일 구조

```
project/
│
├── notebooks/                   # EDA 결과, 임시 코드를 위한 Jupyter 노트북 디렉토리
│ └── (EDA 결과 또는 임시 코드)   # 실제 노트북 파일들
│
├── src/ # 소스 코드의 메인 디렉토리
│ ├── QuestionAnswering/         # 질문 답변 관련 모듈 디렉토리
│ │ ├── tokenizer_wrapper.py     # 토크나이저 래퍼 클래스/함수
│ │ ├── trainer.py               # 모델 훈련 관련 클래스/함수
│ │ └── utils.py                 # QA 관련 유틸리티 함수
│ │
│ ├── Retrieval/                 # 검색 관련 모듈 디렉토리
│ │ ├── cross_encoder.py         # Cross-encoder 모델 관련 코드
│ │ ├── dense_retrieval.py       # 밀집 검색 관련 코드
│ │ ├── sparse_retrieval.py      # 희소 검색 관련 코드
│ │ ├── hybrid_retrieval.py      # 하이브리드 검색 관련 코드
│ │ ├── NegativeSampler.py       # 일반 네거티브 샘플링 클래스
│ │ └── SparseNegativeSampler.py # 희소 검색 기반 네거티브 샘플링 클래스
│ │
│ ├── server/                    # Streamlit 서버 관련 코드 디렉토리
│ │ ├── page/                    # 페이지 구현 클래스 디렉토리
│ │ │ ├── DataEDA.py             # 데이터 EDA 페이지 구현
│ │ │ ├── HomePage.py            # 홈페이지 구현
│ │ │ ├── TokenizerEDA.py        # 토크나이저 분석 페이지 구현
│ │ │ └── trainingDatasetQA.py   # QA 데이터셋 분석 페이지 구현
│ │ │
│ │ └── utils/                   # 서버 유틸리티 디렉토리
│ │ ├── data_loader.py           # 데이터 로딩 관련 함수
│ │ └── Page.py                  # 페이지 관리 클래스
│ │
│ ├── config.py                  # 설정 파일 로드 및 파싱
│ ├── index.py                   # Streamlit 앱의 메인 엔트리 포인트
│ ├── main.py                    # 전체 모델 훈련 및 검증 관리
│ └── preprocess.py              # 텍스트 전처리 함수
│
├── config.yaml                  # 기본 모델 및 데이터 경로 설정
└── dense_encoder_config.yaml    # Dense Encoder 모델 설정
```

## 4. 프로젝트 수행 절차 및 방법

### 4.1 그라운드 룰

1. 팀 Notion의 서버 현황판 활용
2. Git 관련 규칙 (commit convention, branch naming convention 등)
3. 소통 관련 규칙 (상호 존중, 실시간 대화, 데일리 스크럼 등)

### 4.2 전체 프로젝트 수행 과정

1. 프로젝트 기초 강의 수강 (09/30 - 10/09)
2. 베이스라인 코드 분석 및 초기 설정 (10/10 - 10/11)
3. Retrieval 및 Streamlit 서버 초기 구현 (10/14 - 10/17)
4. 모델 개선 및 데이터 증강 실험 (10/18 - 10/21)
5. 모델 탐색 및 최적화 (10/22 - 10/23)

## 5. 프로젝트 상세

### 5.1 협업 툴

- **Notion**: 실험 진행 현황, 서버 사용 현황, 프로젝트 일정 정리 및 실험 기록 정리.
- **Git**: branch, commit convention 설정, issue 기능 활용, Release 기능 사용. 세부적인 버전 관리의 어려움 존재.
- **Zoom**: 데일리 스크럼 및 피어세션을 통해 원활한 소통 유지.

### 5.2 파일 구조

- **초기 구조**: 모든 파일을 `src` 폴더에 넣고, `Retrieval`과 `QuestionAnswering` 폴더로 구분.
- **변경 후**: `src` 폴더에 `server` 폴더 추가, `inference.py`와 `train.py`를 통합하여 `main.py` 구축.

### 5.3 데이터 EDA
#### Streamlit 서버를 활용한 EDA 결과 배포
- **목적**: 팀원이 훈련용 데이터셋을 쉽게 확인할 수 있도록 GPU 서버에 Streamlit 서버를 배포.
- **기능**: 질문과 Context 전문을 보여주는 페이지, 정답 강조 기능, UNK 토큰 강조 기능.

#### 데이터 증강 및 전처리
- **결과**: 한자와 이스케이프 문자 등이 UNK 토큰으로 식별됨.
- **전처리 시도**: 한자 제거 시도, KorQuAD를 활용한 데이터 증강, 특수문자 및 연속 공백 제거.


### 5.4 기능 개발 상세

#### Sparse Embedding
- **변경 사항**: TF-IDF 대신 BM25 적용.
- **라이브러리**: rank_bm25 사용.

#### Dense Embedding
- **모델**: Query Encoder와 Passage Encoder 기반 Bi-Encoder 모델 구현.
- **기법**: Hard Negative Sampling 활용.

#### Cross Encoder
- **목적**: Query와 Passage 사이의 어텐션 값을 계산하여 유사도 랭킹 구함.
- **결과**: BM25보다 높은 Recall@5 성능.

#### Extraction-Based Reader
- **기본 모델**: 기존 베이스라인 코드 모델 사용.

#### Generation-Based Reader
- **구현 이유**: Extraction-based 모델의 한계를 극복하기 위해.
- **기법**: LoRA 기법 적용.

### 5.5 앙상블
다양한 모델 조합을 통한 성능 향상 시도

### 5.6 사용한 모델

#### BERT-base, ELECTRA-base
- `klue/bert-base`, `monologg/koelectra-base-v3-discriminator`

#### DeBERTa-base
- `timpal0l/mdeberta-v3-base-squad2` ([후처리 함수](https://colab.research.google.com/drive/1VMld_ULNAaOsoX39gPFTzmxI9RE6jgDY?usp=sharing)), `kakaobank/kf-deberta-base`

#### RoBERTa-base
- `Dongjin-kr/ko-reranker`, `hongzoh/roberta-large-qa-korquad-v1`

## 6. 팀회고
### 6.1 협업 및 프로젝트 관리
- Notion, Git, Zoom 등 다양한 협업 도구의 효과적 활용으로 원활한 소통과 진행 상황 공유
- GitHub의 이슈, PR, Release 기능을 적극 활용한 버전 관리와 작업 추적 시도

### 6.2 데이터 중심 접근
- Streamlit을 활용한 EDA 서버 구축으로 실시간 데이터 분석과 공유 실현
- UNK 토큰 분석 등 데이터 증강 및 전처리 과정에서의 세밀한 접근 시도

### 6.3 모델 개발 및 실험
- Sparse Retrieval(BM25), Dense Retrieval, Cross Encoder 등 다양한 접근 방식 시도
- Extraction-based와 Generation-based Reader 모델의 비교 분석
- 하이퍼파라미터 튜닝, LoRA 적용 등 성능 최적화 노력

### 6.4 FeedBack
- 리더보드 순위보다 학습과 성장에 초점을 맞춘 프로젝트 진행
- `Generation-based MRC`, `Dense Retrieval`, `Cross Encoder` 등 다양한 접근 방식을 시도하여 `성장`이라는 목표를 이루는 데 도움이 많이 되었음
- 다만, PM의 부재로 인해 체계적인 일정 관리와 명확한 역할 분담의 부족했다고 판단됨.
- 또한 후반부로 갈수록 코드 복잡성 증가하였고 시간 관리 미흡 등으로 인해 실험 기록 및 가설 검증 들을 체계화 하는 것에 어려움을 경험

### 6.5 향후 개선 방향
- PM 역할 도입을 통한 체계적인 일정 관리와 명확한 역할 분담 필요
- 명확한 가설 수립 및 체계적인 검증 과정을 확립하는 것을 넘어서 이를 구체적이게 실시간으로 공유할 Dash Board를 구현할 것
- 코드 리뷰와 정기적인 전체 프로젝트 구조 점검을 통한 코드 복잡성 관리를 할 것
