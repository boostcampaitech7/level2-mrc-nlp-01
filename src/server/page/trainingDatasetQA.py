import streamlit as st
from server.utils.Page import Page
from server.utils.data_loader import load_dataset
from datasets import load_from_disk
from config import Config
from transformers import AutoTokenizer

class TrainingDatasetQAPage(Page):
    page_name = "TrainingDatasetQA"
    alias = "QA 훈련용 데이터셋"
    parent = "Home"
    
    def __init__(self):
        super().__init__()
    
    
    @st.cache_data
    def load_tokenizer(_self):
        config = Config()
        return AutoTokenizer.from_pretrained(config.model.name(), use_fast=True)

    def get_is_unk(_self, context, tokenizer):
        tokenized_context = tokenizer(context)["input_ids"]
        decoded_context = tokenizer.decode(tokenized_context, skip_special_tokens=True)
        
        unspaced_context = context.replace(" ", "")
        unspaced_decoded_context = decoded_context.replace(" ", "")
        
        origin_idx = 0
        decoded_idx = 0
        is_unk = [False for _ in unspaced_context]
        
        # is_unk를 spacing을 다 없앤 상태에서 구하기
        while decoded_idx < len(unspaced_decoded_context):
            if unspaced_context[origin_idx] == unspaced_decoded_context[decoded_idx]:
                decoded_idx += 1
                origin_idx += 1
            else:
                is_unk[origin_idx] = True
                origin_idx += 1
        
        while origin_idx < len(unspaced_context):
            is_unk[origin_idx] = True
            origin_idx += 1
        
        # spacing을 다시 넣어주기
        unspaced_idx = 0
        spaced_idx = 0
        spaced_is_unk = [False for _ in context]
        
        while spaced_idx < len(context):
            if context[spaced_idx] == ' ':
                spaced_idx += 1
                continue
            else:
                spaced_is_unk[spaced_idx] = is_unk[unspaced_idx]
                unspaced_idx += 1
                spaced_idx += 1
        
        return spaced_is_unk

    def get_idx_color(_self, desired, colors, want_to_coloring):
        assert len(colors) == len(want_to_coloring)
        return [desired if now_color is None and wanted else now_color
                for now_color, wanted in zip(colors, want_to_coloring)]
    
    def highlight(_self, context, colors):
        highlighted_context = ""
        skip_word = [" ", "\\", "\n"]
        for idx, color in enumerate(colors):
            if color is None or context[idx] in skip_word:
                highlighted_context += context[idx]
            else:
                highlighted_context += f":{color}[**{context[idx]}**]"
        return highlighted_context

    def load_data(self, data):
        answers = data["answers"]
        answers = [(answers['text'][i], answers['answer_start'][i]) for i in range(len(answers['text']))]
        
        question = data["question"]
        context = data["context"]

        st.write("---\nQ: "+question)
        
        is_answer = [False for _ in context]
        for (text, idx) in answers:
            end_idx = idx + len(text)
            for i in range(idx, end_idx):
                is_answer[i] = True
        
        colors = [None for _ in context]
        colors = self.get_idx_color("red", colors, is_answer)
        
        tokenizer = self.load_tokenizer()
        is_unk = self.get_is_unk(context, tokenizer)
        nonannotated_context = context
        if st.checkbox("UNK 토큰 표시"):
            colors = self.get_idx_color("blue", colors, is_unk)
            nonannotated_colors = [None for _ in context]
            nonannotated_colors = self.get_idx_color("blue", nonannotated_colors, is_unk)
            nonannotated_context = self.highlight(context, nonannotated_colors)
        annotated_context = self.highlight(context, colors)
        nonannotated_context = nonannotated_context.replace("\\n", "\n")
        annotated_context = annotated_context.replace("\\n", "\n")
        
        # # st.header("Context")
        # placeholder = st.empty()
        if st.checkbox("정답 공개"):
            st.write("---")
            st.write(annotated_context)
        else:
            st.write("---")
            st.write(nonannotated_context)
        
    
    def body(self):
        dataset  = load_dataset()
        train_dataset = dataset["train"]
        idx = st.slider("몇 번째 데이터셋을 보시겠습니까?", 0, len(train_dataset)-1, 0)
        input_idx = st.number_input("", value=idx, min_value=0, max_value=len(train_dataset)-1)
        self.load_data(train_dataset[input_idx])
        