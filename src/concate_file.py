import shutil
import os
from datasets import load_dataset, load_from_disk, concatenate_datasets, Sequence, Value

# KorQuAD 불러오기
korquad_dataset = load_dataset("squad_kor_v1")

# 기존 train_dataset 불러오기
train_dataset = load_from_disk("./data/train_dataset/train")
valid_dataset = load_from_disk("./data/train_dataset/validation")

# document_id 필드 제거
train_dataset = train_dataset.remove_columns(['document_id'])
valid_dataset = valid_dataset.remove_columns(['document_id'])

# 'answers' 필드 타입 일치
train_features = train_dataset.features
valid_features = valid_dataset.features

train_features['answers'] = Sequence({
    'answer_start': Value('int64'),
    'text': Value('string')
})

valid_features['answers'] = Sequence({
    'answer_start': Value('int64'),
    'text': Value('string')
})

# 데이터셋 형변환
train_dataset = train_dataset.cast(train_features)
valid_dataset = valid_dataset.cast(train_features)

korquad_train = korquad_dataset['train']
korquad_validation = korquad_dataset['validation']

korquad_features = korquad_train.features
korquad_features['answers'] = Sequence({
    'answer_start': Value('int64'),
    'text': Value('string')
})

korquad_train = korquad_train.cast(korquad_features)
korquad_validation = korquad_validation.cast(korquad_features)

# 최초의 train_dataset 갯수 출력
original_train_size = len(train_dataset)
print(f"Original train dataset size: {original_train_size}")

# 데이터셋 병합
combined_dataset = concatenate_datasets([train_dataset, valid_dataset, korquad_train, korquad_validation])

# 전체 데이터셋을 95:5 비율로 나누기
train_test_split = combined_dataset.train_test_split(test_size=0.05)

# 최종 train dataset 갯수 출력
final_train_size = len(train_test_split['train'])
print(f"Final train dataset size after merging: {final_train_size}")

# 저장 경로 설정 (현재 디렉토리로 설정)
base_path = "."
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "validation")

# 새 디렉토리 생성
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

# 나누어진 데이터셋 저장
train_test_split['train'].save_to_disk(train_path)
train_test_split['test'].save_to_disk(valid_path)

print(f"New train dataset saved to: {train_path}")
print(f"New validation dataset saved to: {valid_path}")
print(f"생성된 train, validation 폴더를 data/train_dataset 내에 있는 train, validation 폴더를 지우고 대체하세요.")