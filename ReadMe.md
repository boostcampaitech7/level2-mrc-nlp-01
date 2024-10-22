## version 설명
- 본 버전은 v3으로 v2와는 다음의 점에서 크게 다르다.
  - Generation-base MRC 가 구현되었다. yaml파일에서 관련 설정을 조정할 수 있다.
  - Dense Encoder를 통한 Dense Retriever가 구현되었다. (단, 굉장히 시간이 오래 걸림)
  - streamlit 서버를 통하여 훈련데이터와 토크나이저에 따른 데이터 EDA가 구현되었다.
  - Evaluation을 진행함에 있어서 기존에 Reader에 대한 성능평가만이 이루어진 것을 수정하여 Retriever를 통해 넘겨 받은 후 Reader가 평가받을 수 있도록 하였다. (Retriever 성능 반영)
  - Cross Encdoer를 추가하였다.
  - bert Encoder 외에 RoBERTa Encoder 또한 동작할 수 있도록 추가하였다.
  - WandB 실험 로깅 기능을 추가하였다.
 

## CLI commnad
    python src/main.py --do_train
    
    python src/main.py --do_eval

    python src/main.py --do_predict

  위 세가지 커맨드는 v2와 동일하고

  이에 덧붙여 
    `--do_mrc` `--do_retrieval` `--do_both` 
  를 진행할 수 있는데
  
  예를 들어 아래와 같다. (리트리버 성능 평가)
  
    python src/main.py --do_retrieval --do_eval

  다만, 현재 에러가 있어 아래와 같은 커맨드는 불가하다.
  
    --do_train --do_both


## 관련 PR 모음
- Feat: Generation-based MRC 구현 by @swbaek97 in #37
- Refactor: dense encoder 설정 변경 by @Now100 in #38
- Feature: streamlit에서 context의 어떤 부분이 UNK 토큰인지 확인할 수 있도록 추가 by @rasauq1122 in #40
- feat: negative sampling 변경 by @Now100 in #44
- fix: Fix negative sampling storage in dense retrieval by @Now100 in #46
- feat: BM25 성능 검증 by @rasauq1122 in #45
- Fix: Evaluation error during generation model training by @swbaek97 in #52
- feat: evaluation시 retriever와 연동해서 진행할 수 있습니다. by @rasauq1122 in #59
- feat: Cross Encoder 추가 by @rasauq1122 in #51
- Feat: RoBERTa Encoder 추가 & Mixed Precision 추가 by @Now100 in #53
- Feat: EDA streamlit 서버 관련하여 기능추가 (데이터셋 길이 보기, 토크나이저에 따른 데이터 EDA) by @hskhyl in #55
- Feat: wandb logging 코드 추가 by @hskhyl in #57
