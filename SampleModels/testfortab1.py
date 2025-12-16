import streamlit as st
import ollama
import subprocess
from pathlib import Path

PLANTUML_JAR = "libs/plantuml.jar"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def render_test1():
    st.title("📊 PUML Diagram Generator")

    user_prompt = st.text_area(
        "Enter system / flow description",
        height=150,
        placeholder="Example: User login flow with authentication service"
    )

    diagram_type = st.selectbox(
        "Select Diagram Type",
        ["Sequence Diagram", "Class Diagram", "Activity Diagram"]
    )

    generate_btn = st.button("🚀 Generate PUML")

    # -----------------------------
    # Generate PUML
    # -----------------------------
    if generate_btn and user_prompt.strip():

        with st.spinner("Generating PUML..."):
            puml_prompt = build_puml_prompt(user_prompt, diagram_type)

            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": puml_prompt}]
            )

            st.session_state["puml_content"] = response["message"]["content"]

        save_and_render()

    # -----------------------------
    # Edit + Save Section
    # -----------------------------
    if "puml_content" in st.session_state:

        st.subheader("📝 Editable PUML")

        edited_puml = st.text_area(
            "Edit PUML and save to regenerate diagram",
            value=st.session_state["puml_content"],
            height=300
        )

        col1, col2 = st.columns(2)

        with col1:
            save_btn = st.button("💾 Save & Regenerate")

        with col2:
            download_btn = st.download_button(
                "📥 Download PUML",
                data=edited_puml,
                file_name="diagram.puml",
                mime="text/plain"
            )

        if save_btn:
            st.session_state["puml_content"] = edited_puml
            save_and_render()

        # Show image if exists
        image_file = OUTPUT_DIR / "diagram.png"
        if image_file.exists():
            st.subheader("🖼️ Diagram Preview")
            st.image(str(image_file), use_container_width=True)


# -----------------------------
# Helper Functions
# -----------------------------
def save_and_render():
    puml_file = OUTPUT_DIR / "diagram.puml"
    puml_file.write_text(st.session_state["puml_content"])

    generate_image(puml_file)


def build_puml_prompt(user_prompt, diagram_type):
    return f"""
Generate a valid PlantUML {diagram_type}.

Rules:
- Start with @startuml
- End with @enduml
- No explanation
- Only PlantUML syntax

System Description:
{user_prompt}
"""


def generate_image(puml_file):
    subprocess.run(
        ["java", "-jar", PLANTUML_JAR, "-tpng", str(puml_file)],
        check=False
    )
