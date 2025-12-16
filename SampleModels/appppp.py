import streamlit as st

from tools.test_case_generator import render_test_case_generator
from tools.test1 import render_test1
from tools.test2 import render_test2

st.set_page_config(page_title="Tester Bot", layout="wide")

st.sidebar.title("🧰 Tester Bot")

# Tool registry (router)
TOOLS = {
    "Test Case Generator": render_test_case_generator,
    "Test1": render_test1,
    "Test2": render_test2
}

selected_tool = st.sidebar.radio(
    "Select Tool",
    list(TOOLS.keys())
)

# 🔥 Dynamic routing (NO if / else)
TOOLS[selected_tool]()
