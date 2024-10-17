## version 설명
- 본 버전은 v2.0.0으로 v1.1.0과는 다음의 점에서 크게 다르다.
  - Dense retriever가 추가되었다. config.yaml에서 dense로 설정하면 실행시 dense_retriever로 실행된다.
  - Sparse retriever 관련하여 TF-IDF 가 아닌, BM25를 활용한다.
  - train.py와 inference.py를 통합하여 main.py로 옮겼다.
  - 덜 구현된 하이퍼 파라미터들을 training_args로 받아 제대로 적용시켰다.
  - EDA 관련 server를 하나 개설했고 해당 서버는 streamlit으로 구현되어 QA 관련 데이터셋을 시각화 및 분석할 수 있도록 하였다.
  - 실험 편이성과 에러 방지를 위해 기존 CLI commnad 를 다음 셋으로 간략화했다.
    
    `python src/main.py --do_train`
    
    `python src/main.py --do_eval`

    `python src/main.py --do_predict`


## 관련 PR 모음
* feat: TF-IDF를 이용한 Sparse Retrieval의 성능 측정 by @rasauq1122 in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/22
* fix : top_k 설정 안 되던 버그 수정 by @hskhyl in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/27
* Feat : change TF-IDF to BM25 by @hskhyl in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/23
* refactor: streamlit 폴더 구조 개편 by @rasauq1122 in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/30
* refactor: config 사용법 정상화 by @rasauq1122 in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/33
* Refactor/Feat: CLI 커맨드 통일화 및 train.py 와 inference.py를 main.py로 합침. by @hskhyl in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/29
* Feat: Dense Retrieval 구현 by @Now100 in https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/35


**Full Changelog**: https://github.com/boostcampaitech7/level2-mrc-nlp-01/compare/v1.1.0...v2.0.0
