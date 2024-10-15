import os
import importlib
import collections

import streamlit as st

def load_pages():
    
    pwd = os.path.dirname(__file__)
    pages_dir = os.path.join(pwd, 'server/page')
    pages_files = os.listdir(pages_dir)
    
    root = None
    file_tree = collections.defaultdict(list)
    name_to_page = {}
    
    for pages_file in pages_files:
        if pages_file.endswith('.py'):
            module_name = pages_file[:-3]
            module = importlib.import_module(f'server.page.{module_name}')
            
            for attr in dir(module):
                obj_by_attr = getattr(module, attr)
                if hasattr(obj_by_attr, "page_name") and obj_by_attr.page_name is not None:
                    if obj_by_attr.parent is None:
                        if root is None:
                            root = obj_by_attr.page_name
                        else:
                            raise ValueError(f"multiple roots: {root} is root, but {obj_by_attr.page_name} is also root")
                    else:
                        file_tree[obj_by_attr.parent].append(obj_by_attr.page_name)
                    name_to_page[obj_by_attr.page_name] = obj_by_attr()
    
    assert root is not None, "no root page found"
    
    return root, file_tree, name_to_page

def render_sidebar(page_name, root, file_tree, name_to_page):
    placeholder = st.sidebar.empty()
    if page_name == root:
        placeholder.title("Home")
    else:
        if placeholder.button("상위 페이지로 돌아가기"):
            st.session_state.page = name_to_page[page_name].parent
            
    if file_tree[page_name] != []:
        page_list = [page_name] + file_tree[page_name]
        page_alias = [name_to_page[page].alias for page in page_list]
        alias_dict = {alias: page for alias, page in zip(page_alias, page_list)}
        next_page = placeholder.radio("하위 페이지", tuple(page_alias))
        st.session_state.page = alias_dict[next_page]
    
    return placeholder

def main():
    root, file_tree, name_to_page = load_pages()
    
    if "page" not in st.session_state:
        st.session_state.page = root
    
    prev_page = st.session_state.page
    placeholder = render_sidebar(st.session_state.page, root, file_tree, name_to_page)
    name_to_page[st.session_state.page].render()
    if prev_page != st.session_state.page:
        placeholder.empty()
        placeholder = render_sidebar(st.session_state.page, root, file_tree, name_to_page)
    
    # print(st.session_state.page)
    
if __name__ == "__main__":
    main()