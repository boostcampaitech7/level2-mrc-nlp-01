import streamlit as st
from server.utils.Page import Page
from server.utils.data_loader import load_dataset
from datasets import load_from_disk
from config import Config
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataEDA(Page):
    page_name = "DataEDA"
    alias = "context, answer 길이 시각화"
    parent = "Home"

    def __init__(self):
        super().__init__()


    def make_data_to_df(self, data):  # 데이터셋이 아닌 변환된 데이터 프레임 캐싱
        # 필요한 컬럼만 선택
        temp_df = {
            'title': [item['title'] for item in data],
            'context': [item['context'] for item in data],
            'id': [item['id'] for item in data],
            'answers_text': [item['answers']['text'][0] if item['answers']['text'] else '' for item in data]
        }
        # 데이터 프레임 변환
        data_df = pd.DataFrame(temp_df)
        # context 길이 추가
        data_df['context_length'] = data_df['context'].apply(len)
        data_df['answer_length'] = data_df['answers_text'].apply(len)

        return data_df


    def make_df_sorted_by_context_len(self, data):  # 언더스코어 추가
        data = data.sort_values(by='context_length')
        return data

    def body(self):
        dataset = load_dataset()
        train_data = dataset['train']
        valid_data = dataset['validation']
        train_df = self.make_data_to_df(train_data)
        valid_df = self.make_data_to_df(valid_data)
        train_df = self.make_df_sorted_by_context_len(train_df)
        valid_df = self.make_df_sorted_by_context_len(valid_df)

        # 히스토그램
        st.subheader('1.1 TrainDataset context 시각화(histogram)')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(train_df['context_length'], bins=100, kde=True, ax=ax)
        ax.set_xlim(512, 2059)
        ax.set_xticks([512, 2059])
        ax.set_title('Context Length Distribution')
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # 박스 플롯
        st.subheader('1.2 TrainDataset context 시각화(Box Plot)')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=train_df['context_length'], ax=ax)
        ax.set_xlim(512, 2059)
        ax.set_xticks([512, 2059])
        ax.set_title('Context Length Box Plot')
        ax.set_xlabel('Context Length')
        st.pyplot(fig)
        
        # valid 히스토그램
        st.subheader('1.3 TrainDataset answer 시각화(histogram)')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(x=train_df['answer_length'], ax=ax)
        ax.set_xlim(1, 83)
        ax.set_xticks([1, 83])
        ax.set_title('Context Length Box Plot')
        ax.set_xlabel('Context Length')
        st.pyplot(fig)
        
        # valid box plot
        st.subheader('1.3 TrainDataset answer 시각화(Box Plot)')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=train_df['answer_length'], ax=ax)
        ax.set_xlim(1, 83)
        ax.set_xticks([1, 83])
        ax.set_title('Context Length Box Plot')
        ax.set_xlabel('Context Length')
        st.pyplot(fig)
