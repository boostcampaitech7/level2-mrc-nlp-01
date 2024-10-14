import streamlit as st
from page.Page import Page
from datasets import load_from_disk

class TrainingDatasetQAPage(Page):
    page_name = "TrainingDatasetQA"
    alias = "QA 훈련용 데이터셋"
    parent = "Home"
    
    def __init__(self):
        super().__init__()
    
    @st.cache_resource
    def load_dataset(_self):
        return load_from_disk("./data/train_dataset")
    
    def load_data(self, data):
        answers = data["answers"]
        answers = [(answers['text'][i], answers['answer_start'][i]) for i in range(len(answers['text']))]
        
        question = data["question"]
        context = data["context"]
        st.markdown(
            """
            <style>
            .stCode {color: red !important;}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        highlighted_context = context
        for (text, idx) in answers:
            end_idx = idx + len(text)
            highlighted_context = f"{highlighted_context[:idx]}:red[**{text}**]{highlighted_context[end_idx:]}"
        
        context = context.replace("\\n", "\n")
        highlighted_context = highlighted_context.replace("\\n", "\n")
        
        st.write("---\nQ: "+question)
        
        # # st.header("Context")
        # placeholder = st.empty()
        if st.checkbox("정답 공개"):
            st.write("---")
            st.write(highlighted_context)
        else:
            st.write("---")
            st.write(context)
        
    
    def body(self):
        dataset  = self.load_dataset()
        train_dataset = dataset["train"]
        idx = st.slider("몇 번째 데이터셋을 보시겠습니까?", 0, len(train_dataset)-1, 0)
        input_idx = st.number_input("", value=idx, min_value=0, max_value=len(train_dataset)-1)
        self.load_data(train_dataset[input_idx])
        