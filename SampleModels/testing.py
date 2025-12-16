import streamlit as st
import ollama
import pandas as pd
import json
from io import BytesIO

# ---------------- Page Config ----------------
st.set_page_config(page_title="PUML Test Case Generator", layout="wide")

st.title("🧪 PUML Test Case Generator")
st.caption("Generate → Edit → Save → Convert / Download")

# ---------------- Session State Init ----------------
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

if "df" not in st.session_state:
    st.session_state.df = None

if "edited_df" not in st.session_state:
    st.session_state.edited_df = None

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload PUML File", type=["puml", "txt"])

# ---------------- Generate ----------------
if uploaded_file and st.button("🚀 Generate Test Cases"):

    puml_content = uploaded_file.read().decode("utf-8")

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
- Follow proper numbering
- No extra text

PlantUML content:
{puml_content}
"""

    with st.spinner("Generating test cases using Llama 3.2..."):
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

    try:
        testcases = json.loads(response["message"]["content"])
        st.session_state.df = pd.DataFrame(testcases)
        st.session_state.edit_mode = True
        st.session_state.edited_df = None
        st.success("Test cases generated successfully")

    except Exception:
        st.error("Invalid JSON received from LLM")
        st.code(response["message"]["content"])
        st.stop()

# ---------------- Editable Table ----------------
if st.session_state.df is not None:

    df = st.session_state.df.copy()

    # Convert list → multiline string for editing
    df["procedure"] = df["procedure"].apply(lambda x: "\n".join(x))
    df["expected_output"] = df["expected_output"].apply(lambda x: "\n".join(x))

    st.subheader("✏️ Review & Edit Test Cases")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor"
    )

    if st.button("💾 Save Changes"):
        st.session_state.edited_df = edited_df
        st.session_state.edit_mode = False
        st.success("Changes saved. Convert and Download are now enabled.")

# ---------------- Convert Button ----------------
st.markdown("### 🔄 Convert")

st.button(
    "🔄 Convert",
    disabled=st.session_state.edit_mode
)

if st.session_state.edit_mode:
    st.caption("Save changes to enable Convert")

# ---------------- Download ----------------
if st.session_state.edited_df is not None:

    final_df = st.session_state.edited_df.copy()

    # Convert multiline string → list (optional)
    final_df["procedure"] = final_df["procedure"].apply(lambda x: x.split("\n"))
    final_df["expected_output"] = final_df["expected_output"].apply(lambda x: x.split("\n"))

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Test Cases")

    st.download_button(
        "⬇️ Download Updated Excel",
        data=output.getvalue(),
        file_name="Updated_Test_Cases.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=st.session_state.edit_mode
    )

    if st.session_state.edit_mode:
        st.caption("Save changes to enable Download")
