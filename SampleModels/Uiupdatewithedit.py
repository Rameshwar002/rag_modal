import streamlit as st
import pandas as pd
import re

# =====================================================
# OPTIONAL LLM (OLLAMA)
# =====================================================
try:
    import ollama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# =====================================================
# INTERNAL BDD LOGIC (RULE BASED)
# =====================================================

def _parse_steps(text):
    if pd.isna(text):
        return []
    lines = text.replace("\r", "").split("\n")
    return [l.strip().lstrip("0123456789. ") for l in lines if l.strip()]


def _find_placeholders(text):
    if pd.isna(text):
        return []
    return re.findall(r"<(.+?)>", text)


def generate_bdd(df):
    bdd = ""

    for feature, fdf in df.groupby("test_case_mapping"):
        tags = fdf["region"].dropna().unique()
        if len(tags) > 0:
            bdd += " ".join(tags) + "\n"

        bdd += f"Feature: {feature}\n\n"

        unique_pre = fdf["pre_requisite"].dropna().unique()
        background_used = False

        if len(unique_pre) == 1:
            bdd += "  Background:\n"
            bdd += f"    Given {unique_pre[0]}\n\n"
            background_used = True

        for desc, sdf in fdf.groupby("Description"):
            sample = sdf.iloc[0]

            placeholders = set(
                _find_placeholders(sample["procedure"]) +
                _find_placeholders(sample["expected_output"])
            )

            is_outline = len(placeholders) > 0 and len(sdf) > 1
            scenario_type = "Scenario Outline" if is_outline else "Scenario"

            bdd += f"  {scenario_type}: {desc}\n"

            if not background_used and pd.notna(sample["pre_requisite"]):
                bdd += f"    Given {sample['pre_requisite']}\n"

            steps = _parse_steps(sample["procedure"])
            if steps:
                bdd += f"    When {steps[0]}\n"
                for step in steps[1:]:
                    bdd += f"    And {step}\n"

            if pd.notna(sample["expected_output"]):
                bdd += f"    Then {sample['expected_output']}\n"

            if is_outline:
                bdd += "\n    Examples:\n"
                headers = list(placeholders)
                bdd += "      | " + " | ".join(headers) + " |\n"
                for _ in range(len(sdf)):
                    bdd += "      | " + " | ".join(["TBD"] * len(headers)) + " |\n"

            bdd += "\n"

    return bdd


# =====================================================
# OPTIONAL LLM BDD ENHANCEMENT
# =====================================================

def enhance_bdd_with_llm(bdd_text, model="llama3"):
    if not LLM_AVAILABLE:
        return bdd_text

    prompt = f"""
You are a BDD expert.
Improve wording only.
Do NOT change keywords or structure.

{bdd_text}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# =====================================================
# LLM → JAVA STEP DEFINITIONS
# =====================================================

def generate_java_with_llm(bdd_text, model="llama3"):
    if not LLM_AVAILABLE:
        return "// LLM not available"

    prompt = f"""
You are a senior Java automation engineer.

Convert the following BDD feature file into Java Step Definitions.

Rules:
- Use Cucumber annotations
- Class name: StepDefinitions
- Generate only method stubs
- No explanations or markdown
- Output ONLY valid Java code

BDD:
{bdd_text}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# =====================================================
# 🔥 MAIN STREAMLIT UI
# =====================================================

def render_bdd_generator_ui():
    st.title("🧪 BDD → Java Generator (LLM Powered)")
    st.caption("Excel → BDD → Edit → Save → Java")

    # Session state
    if "bdd_text" not in st.session_state:
        st.session_state.bdd_text = ""
    if "saved" not in st.session_state:
        st.session_state.saved = False

    uploaded_file = st.file_uploader(
        "📂 Upload Test Case Excel",
        type=["xlsx"]
    )

    use_llm_bdd = st.checkbox(
        "✨ Enhance BDD using LLM",
        disabled=not LLM_AVAILABLE
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        required_cols = {
            "test_case_mapping",
            "Description",
            "pre_requisite",
            "procedure",
            "expected_output",
            "region"
        }

        if not required_cols.issubset(df.columns):
            st.error("❌ Invalid Excel format")
            return

        st.success("✅ Excel validated")

        if st.button("🚀 Generate BDD"):
            with st.spinner("Generating BDD..."):
                bdd = generate_bdd(df)
                if use_llm_bdd:
                    bdd = enhance_bdd_with_llm(bdd)

            st.session_state.bdd_text = bdd
            st.session_state.saved = False

    # ---------- EDIT MODE ----------
    if st.session_state.bdd_text:
        st.subheader("✏️ Edit BDD Output")

        edited_bdd = st.text_area(
            "Editable BDD",
            st.session_state.bdd_text,
            height=400
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save"):
                st.session_state.bdd_text = edited_bdd
                st.session_state.saved = True
                st.success("BDD saved successfully")

        with col2:
            st.download_button(
                "⬇️ Download .feature",
                edited_bdd,
                file_name="generated.feature",
                mime="text/plain"
            )

    # ---------- JAVA GENERATION ----------
    if st.session_state.saved:
        st.subheader("☕ Java Step Definitions")

        if st.button("🤖 Generate Java using LLM"):
            with st.spinner("Generating Java..."):
                java_code = generate_java_with_llm(
                    st.session_state.bdd_text
                )

            st.code(java_code, language="java")

            st.download_button(
                "⬇️ Download StepDefinitions.java",
                java_code,
                file_name="StepDefinitions.java",
                mime="text/plain"
            )


# =====================================================
# RUN APP
# =====================================================

render_bdd_generator_ui()
