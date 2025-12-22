import streamlit as st
import pandas as pd
import re

# Optional LLM (Ollama)
try:
    import ollama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# =====================================================
# INTERNAL LOGIC (RULE BASED)
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


def _generate_bdd_rule_based(df):
    bdd = ""

    for feature, fdf in df.groupby("test_case_mapping"):
        # Tags
        tags = fdf["region"].dropna().unique()
        if len(tags) > 0:
            bdd += " ".join(tags) + "\n"

        bdd += f"Feature: {feature}\n\n"

        # ---------- BACKGROUND ----------
        unique_pre = fdf["pre_requisite"].dropna().unique()
        background_used = False

        if len(unique_pre) == 1:
            bdd += "  Background:\n"
            bdd += f"    Given {unique_pre[0]}\n\n"
            background_used = True

        # ---------- SCENARIOS ----------
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

            # ---------- EXAMPLES ----------
            if is_outline:
                bdd += "\n    Examples:\n"
                headers = list(placeholders)
                bdd += "      | " + " | ".join(headers) + " |\n"
                for _ in range(len(sdf)):
                    bdd += "      | " + " | ".join(["TBD"] * len(headers)) + " |\n"

            bdd += "\n"

    return bdd


def _enhance_with_llm(bdd_text, model):
    if not LLM_AVAILABLE:
        return bdd_text

    prompt = f"""
You are a BDD expert.
Improve wording only.
Do NOT change structure, tags, or keywords.

{bdd_text}
"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# =====================================================
# 🔥 ONE MAIN UI FUNCTION (CALL THIS)
# =====================================================

def render_bdd_generator_ui():
    st.header("🧪 BDD Feature Generator")
    st.caption("Excel → Rule-based / LLM-enhanced BDD")

    uploaded_file = st.file_uploader(
        "📂 Upload Test Case Excel",
        type=["xlsx"],
        key="bdd_excel"
    )

    use_llm = st.checkbox("✨ Enhance using LLM", disabled=not LLM_AVAILABLE)
    llm_model = "llama3"

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
            st.error("❌ Excel columns do not match expected test case format")
            return

        st.success("✅ Excel validated")

        # ---------- PROMPT BUTTON ----------
        if st.button("🚀 Generate BDD Feature File"):
            with st.spinner("Generating BDD..."):
                bdd_text = _generate_bdd_rule_based(df)

                if use_llm:
                    bdd_text = _enhance_with_llm(bdd_text, llm_model)

            st.subheader("📄 Generated BDD Preview")
            st.code(bdd_text, language="gherkin")

            st.download_button(
                "⬇️ Download .feature file",
                bdd_text,
                file_name="generated.feature",
                mime="text/plain"
            )
        st.download_button(
            "⬇️ Download .feature file",
            bdd_text,
            "generated.feature",
            "text/plain"
        )
