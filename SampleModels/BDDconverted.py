import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Advanced BDD Generator", layout="centered")
st.title("🧪 Advanced BDD Generator")
st.caption("Background + Scenario Outline supported")

uploaded_file = st.file_uploader("Upload Test Case Excel", type=["xlsx"])


def parse_steps(text):
    if pd.isna(text):
        return []
    lines = text.replace("\r", "").split("\n")
    return [l.strip().lstrip("0123456789. ") for l in lines if l.strip()]


def find_placeholders(text):
    if pd.isna(text):
        return []
    return re.findall(r"<(.+?)>", text)


def generate_bdd(df):
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
                find_placeholders(sample["procedure"]) +
                find_placeholders(sample["expected_output"])
            )

            is_outline = len(placeholders) > 0 and len(sdf) > 1

            if is_outline:
                bdd += f"  Scenario Outline: {desc}\n"
            else:
                bdd += f"  Scenario: {desc}\n"

            if not background_used and pd.notna(sample["pre_requisite"]):
                bdd += f"    Given {sample['pre_requisite']}\n"

            steps = parse_steps(sample["procedure"])
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

                for _ in sdf.itertuples():
                    values = ["TBD" for _ in headers]  # can be extended
                    bdd += "      | " + " | ".join(values) + " |\n"

            bdd += "\n"

    return bdd


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
        st.error("Missing required columns")
    else:
        bdd_text = generate_bdd(df)

        st.subheader("📄 Generated BDD Feature")
        st.code(bdd_text, language="gherkin")

        st.download_button(
            "⬇️ Download .feature file",
            bdd_text,
            "generated.feature",
            "text/plain"
        )
