import streamlit as st
from server.utils.Page import Page
from server.utils.data_loader import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class TokenizerEDA(Page):
    page_name = "TokenizerEDA"
    alias = "Tokenizer에 따른 <unk>토큰 갯수 분석"
    parent = "Home"

    def __init__(self):
        super().__init__()

    # 데이터를 DataFrame으로 변환하는 함수
    def make_data_to_df(self, data):
        # 필요한 데이터를 temp_df에 저장해둠 (title, context, id, answers_text)
        temp_df = {
            'title': [item['title'] for item in data],  # title 컬럼에 데이터 추가
            'context': [item['context'] for item in data],  # context 컬럼에 데이터 추가
            'id': [item['id'] for item in data],  # id 컬럼에 데이터 추가
            'answers_text': [item['answers']['text'][0] if item['answers']['text'] else '' for item in data]  # answers 텍스트 중 첫번째를 사용
        }
        # temp_df로 데이터프레임 생성
        data_df = pd.DataFrame(temp_df)
        # context의 길이 계산 후 새로운 컬럼 'context_length'에 저장
        data_df['context_length'] = data_df['context'].apply(len)
        # answers_text의 길이 계산 후 'answer_length'에 저장
        data_df['answer_length'] = data_df['answers_text'].apply(len)

        return data_df

    # context 길이에 따라 데이터를 정렬하는 함수
    def make_df_sorted_by_context_len(self, data):
        data = data.sort_values(by='context_length')  # context_length 기준으로 정렬
        return data

    # Unknown Token 비율을 계산하는 함수
    def calculate_unknown_token_ratio(self, tokenizer, contexts):
        unk_token_id = tokenizer.unk_token_id  # UNK 토큰의 ID 가져오기
        unk_ratios = []  # 각 문장의 UNK 토큰 비율을 저장할 리스트
        original_unk_tokens = []  # UNK로 변환된 원래의 토큰들을 저장할 리스트
        total_unk_count = 0  # 전체 문장에 있는 UNK 토큰 개수
        total_token_count = 0  # 전체 문장의 토큰 개수
        
        for context in contexts:
            # 문장을 토큰화하면서 각 토큰의 원래 위치를 저장 (offset_mapping)
            encoding = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
            tokens = encoding['input_ids']  # 토큰화된 토큰 ID 리스트
            offsets = encoding['offset_mapping']  # 각 토큰의 원래 문장 내 위치

            # 원본 문장은 단어 단위로 분할
            original_tokens = context.split()

            # UNK 토큰의 개수와 전체 토큰 개수 계산
            unk_count = tokens.count(unk_token_id)
            total_tokens = len(tokens)
            
            # 현재 문장의 UNK 비율 계산
            unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0
            unk_ratios.append(unk_ratio)  # 비율을 리스트에 저장

            # UNK로 변환된 원래의 단어들을 저장
            original_unk = []
            for token_id, offset in zip(tokens, offsets):
                if token_id == unk_token_id:
                    # offset을 사용해 원래 문장에서 UNK로 변환된 단어 찾기
                    start, end = offset
                    original_word = context[start:end]
                    original_unk.append(original_word)
            
            original_unk_tokens.append(original_unk)  # 원래 UNK였던 단어들을 리스트에 저장

            # 전체 UNK 토큰 개수와 전체 토큰 개수를 계속 더해줌
            total_unk_count += unk_count
            total_token_count += total_tokens
        
        # 전체 문장에 대한 총 UNK 비율 계산
        overall_unk_ratio = total_unk_count / total_token_count if total_token_count > 0 else 0
        
        return unk_ratios, original_unk_tokens, overall_unk_ratio

    # Streamlit 페이지의 주요 내용
    def body(self):
        # 사용자가 토크나이저를 입력할 수 있게 함 (기본값은 'klue/bert-base')
        tokenizer_input = st.text_input("Enter tokenizer (e.g., klue/bert-base):", value="klue/bert-base")

        if tokenizer_input:
            # 입력된 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_input)
            dataset = load_dataset()  # 데이터셋 로드
            train_data = dataset['train']
            
            # 데이터를 DataFrame으로 변환
            train_df = self.make_data_to_df(train_data)
            
            # 각 문장의 UNK 비율과 UNK로 변환된 원래 단어 계산
            unk_ratios, original_unk_tokens, overall_unk_ratio = self.calculate_unknown_token_ratio(tokenizer, train_df['context'])
            train_df['unk_token_ratio'] = unk_ratios  # 각 문장의 UNK 비율을 데이터프레임에 추가
            train_df['original_unk_tokens'] = original_unk_tokens  # UNK로 변환된 원래 단어들을 추가
            
            # DataFrame을 화면에 출력 (context, unk_token_ratio, original_unk_tokens)
            st.write(train_df[['context', 'unk_token_ratio', 'original_unk_tokens']])

            # 전체 UNK 비율을 화면에 출력
            st.subheader(f"Overall Unknown Token Ratio: {overall_unk_ratio:.4f}")

            # UNK 비율 분포를 시각화
            st.subheader("Unknown Token Ratio Distribution")
            sns.histplot(unk_ratios, bins=20, kde=True)  # 히스토그램 그리기
            plt.xlabel('Unknown Token Ratio')  # x축 이름 지정
            plt.ylabel('Frequency')  # y축 이름 지정
            st.pyplot(plt)  # 그래프를 Streamlit에 출력
