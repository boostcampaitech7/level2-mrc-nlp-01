import streamlit as st
from page.Page import Page

class HomePage(Page):
    page_name = "Home"
    alias = "홈페이지"
    parent = None
    
    def __init__(self):
        super().__init__()
        
    def header(self):
        st.title(self.alias)
        
    def body(self):
        st.write(
            "왼쪽 사이드바에서 페이지를 선택하세요!"
        )
    