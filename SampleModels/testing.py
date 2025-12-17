import streamlit as st
import ollama
import pandas as pd
import json
from io import BytesIO


# =====================================================
# MAIN UI
# =====================================================
def render_test_case_generator():

    st.title("🧪 PUML Test Case Generator")
    st.caption("Generate → Edit → Convert → Download")

    # ----------------------------
    # Session State Init
    # ----------------------------
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("edited_df", None)
    st.session_state.setdefault("retry", 0)
    st.session_state.setdefault("converted", False)

    uploaded_file = st.file_uploader(
        "Upload PUML File",
        type=["puml", "txt"]
    )

    # ----------------------------
    # Generate
    # ----------------------------
    if uploaded_file and st.button("🚀 Generate Test Cases"):
        puml_content = uploaded_file.read().decode("utf-8")
        generate_test_cases(puml_content)

    # ----------------------------
    # Editable Table
    # ----------------------------
    if st.session_state.df is not None:

        df = st.session_state.df.copy()

        # Convert list → multiline text
        df["procedure"] = df["procedure"].apply(
            lambda x: "\n".join(x) if isinstance(x, list) else ""
        )
        df["expected_output"] = df["expected_output"].apply(
            lambda x: "\n".join(x) if isinstance(x, list) else ""
        )

        st.subheader("✏️ Review & Edit Test Cases")

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="testcase_editor"
        )

        if st.button("💾 Save Changes"):
            st.session_state.edited_df = edited_df
            st.session_state.converted = False
            st.success("✅ Changes saved")

    # ----------------------------
    # Convert Section
    # ----------------------------
    if st.session_state.edited_df is not None:

        st.markdown("### 🔄 Convert")

        if st.button("🔄 Convert to Excel"):
            st.session_state.converted = True
            st.success("✅ Conversion ready")

    # ----------------------------
    # Download Section
    # ----------------------------
    if st.session_state.converted:

        final_df = st.session_state.edited_df.copy()

        # Convert multiline → list
        final_df["procedure"] = final_df["procedure"].apply(
            lambda x: [i.strip() for i in x.split("\n") if i.strip()]
        )
        final_df["expected_output"] = final_df["expected_output"].apply(
            lambda x: [i.strip() for i in x.split("\n") if i.strip()]
        )

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Test_Cases")

        st.download_button(
            "⬇️ Download Excel",
            data=output.getvalue(),
            file_name="Test_Cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =====================================================
# LLM GENERATION (AUTO RETRY)
# =====================================================
def generate_test_cases(puml_content):

    prompt = f"""
Generate at least 5 scenario-based test cases from the PlantUML diagram.

Return ONLY valid JSON.

Each test case must follow this structure:
[
  {{
    "test_case_no": "TC_RHL_AB01_01",
    "test_case_mapping": "RHL_01",
    "description": "",
    "pre_requisite": "",
    "procedure": [],
    "expected_output": [],
    "region": "EU"
  }}
]

Rules:
- Region must be EU, NNA, or JPN
- procedure and expected_output must be arrays
- No extra text
- No markdown

PlantUML content:
{puml_content}
"""

    with st.spinner("🤖 Generating test cases using Llama 3.2..."):
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

    raw_output = response["message"]["content"]

    try:
        data = json.loads(raw_output)

        st.session_state.df = pd.DataFrame(data)
        st.session_state.edited_df = None
        st.session_state.converted = False
        st.session_state.retry = 0

        st.success("✅ Test cases generated successfully")

    except Exception:
        if st.session_state.retry < 1:
            st.session_state.retry += 1
            st.warning("⚠️ Invalid JSON. Auto-regenerating...")
            generate_test_cases(puml_content)
        else:
            st.error("❌ Invalid JSON from LLM")
            st.code(raw_output)
            st.session_state.retry = 0
