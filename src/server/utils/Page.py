import streamlit as st

class Page:
    page_name = None
    alias = page_name
    parent = None
    
    def __init__(self):
        pass
    
    def header(self):
        st.title(self.alias)
                
    def body(self):
        pass
        
    def footer(self):
        pass    
    
    def render(self):
        self.header()
        self.body()
        self.footer()
        
    def render_sidebar(self, root, file_tree, name_to_page):
        if self.page_name == root:
            st.sidebar.title("Home")
        else:
            if st.sidebar.button("상위 페이지로 돌아가기"):
                return name_to_page[self.page_name].parent
                
        if file_tree[self.page_name] != []:
            next_page = st.sidebar.radio("하위 페이지", tuple([self.page_name] + file_tree[self.page_name]))
            return next_page

        return root
        # print(self.page_name, "sidebar rendered done")