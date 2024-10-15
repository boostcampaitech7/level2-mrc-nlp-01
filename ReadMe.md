## version 설명
- 본 버전은 version_0.0으로 가정한 기본 제공된 BaseLineCode에서 준성님이 재구조화 시킨 version_1.0을 기반으로 작동하며
- 해당 버전에 있어 https://github.com/boostcampaitech7/level2-mrc-nlp-01/pull/10 에 기록된 바, `Sparse_retrieval.py` 를 다소 수정하여 retriever 성능 평가 기능을 정상 작동시킴.
- `requirements.txt` 추가
- 코드 수정: train.py
    ```
    from datasets import load_from_disk, load_metric 
    ```
    위 코드를
    ```
    from datasets import load_from_disk
    from evaluate import load as load_metric
    ```
    위와 같이 변경
- 코드 수정: config.py
    ```
    class Config:
    def __init__(self, config_dict=None, path='./config.yaml'):
    ```
    위 부분을
    ```
    class Config:
    def __init__(self, config_dict=None, path='../config.yaml'):
    ```
    와 같이 현재 파일구조와 맞춤.

- 코드 수정: config.yaml
    경로 관련하여 
    CLI에서 실행시킬 `train.py`를
    `/data/ephemeral/home/project/version_1.1/src/train.py` 와 같이 입력시키면 `permission denied` 문제가 있어서
    그냥 터미널을 src 폴더 들어가서 실행시키는 것으로 하고 config.yaml의
    `./data/train_dataset` 라고 되어있던 부분을
    `../data/train_dataset` 이렇게 바꿈.
    원래대로 `python train.py --output_dir ./models/train_dataset --do_train`
    명령하면 이제 됨.

- 기능 추가: `--testing`
    관련하여 전체 데이터셋을 사용하지 않고 testing 할 수 있도록 argument parser 추가
    - `train.py`와 `inference.py`, `sparse_retrieval.py`에 해당 부분이 적용될 수 있도록 코드 수정하였고
    - 구체적으로는 datasets, wikipedia_documents 불러오는 부분을 1%만 불러오게끔 수정함.

- 코드 수정: `.gitignore`에 `__pycache__` 추가함.


## 구체적인 적용점 (GitHub Issue)
- Ljb issue 09 #10 

# 실행 방법
- train의 경우
    python train.py --output_dir ./models/train_dataset --do_train `--testing`(선택)
- eval의 경우
    python train.py --output_dir ./outputs/train_dataset  --do_eval `--testing`(선택)
- inference의 경우
    python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset  --do_predict `--testing`(선택)


- 기능 추가: `sparseDense_retrieval.py`
    - 기존의 `sparse_retrieval.py` 와 `dense_retrieval.py` 에서 두 가지 방법을 모두 사용하여 결과를 출력하는 코드 추가
    - 두 가지 방법의 결과를 합쳐서 출력하는 코드 추가
    - 작동 방식
        - 먼저 두 가지 방법의 임베딩을 모두 준비
        - 두 가지 방법의 임베딩을 가중치(ratid)를 적용하여 최종 점수를 계산
        - 최종 점수를 기준으로 정렬하여 출력

    - run 메소드 추가
        - 기존에 있던 sparse_retrieval.py의 run 메소드 참조
        - 두 가지 방법의 임베딩을 모두 준비
        - faiss를 사용 여부에 따라 임베딩 방법을 선택
        - 예측 모드와 평가 모드에 따라 출력 형식을 선택
        - 최종적으로 결합된 결과를 포함한 데이터셋 반환
    - 참고
        - faiss관련 코드의 경우 상의 후 수정 필요

