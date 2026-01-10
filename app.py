import streamlit as st
import os
from pathlib import Path

from docsentinel2.ingestion import load_document, load_xlsx
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.diff_engine_onnx import DiffEngineONNX
from docsentinel2.visual_diff import run_visual_diff
from docsentinel2.table_diff import diff_workbooks
from docsentinel2.report import generate_report

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="DocSentinel",
    layout="wide",
    page_icon="🕵️‍♂️"
)

st.title("🕵️‍♂️ DocSentinel — Multi-Modal Document Change Detector")


uploaded_old = st.file_uploader("Upload OLD document", type=["pdf", "xlsx"])
uploaded_new = st.file_uploader("Upload NEW document", type=["pdf", "xlsx"])

if uploaded_old and uploaded_new:
    ext_old = os.path.splitext(uploaded_old.name)[1].lower()
    ext_new = os.path.splitext(uploaded_new.name)[1].lower()

    if ext_old != ext_new:
        st.error("❌ Both documents must be of the same type!")
    else:
        # Save temporarily
        old_path = OUTPUT_DIR / "tmp_old" / uploaded_old.name
        new_path = OUTPUT_DIR / "tmp_new" / uploaded_new.name
        old_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(old_path, "wb") as f:
            f.write(uploaded_old.read())
        with open(new_path, "wb") as f:
            f.write(uploaded_new.read())

        run_button = st.button("🚀 Run Comparison")

        if run_button:
            st.info("🧠 Running DocSentinel Pipeline… Please wait ⏳")

            all_changes = []

            embed_model = EmbeddingModel()
            aligner = SemanticAligner(embed_model)
            engine = DiffEngineONNX(embedder=embed_model, aligner=aligner)

            if ext_old == ".pdf":
                text_changes = engine.detect_changes(
                    load_document(old_path),
                    load_document(new_path)
                )
                visual_changes = run_visual_diff(str(old_path), str(new_path))
                all_changes.extend(text_changes + visual_changes)

            else:
                table_changes = diff_workbooks(
                    load_xlsx(old_path),
                    load_xlsx(new_path),
                    embed_model
                )
                all_changes.extend(table_changes)

            # Filter results
            real_changes = [
                c for c in all_changes
                if c.get("label") not in ["NO_CHANGE", "MINOR_CHANGE"]
            ]

            if not real_changes:
                st.success("🎉 No meaningful changes detected!")
            else:
                st.warning(f"Detected {len(real_changes)} changes:")

                for i, ch in enumerate(real_changes, 1):
                    with st.expander(f"Change #{i}: {ch.get('label', ch.get('type'))}"):
                        st.write(ch)
                        if "highlight_path" in ch and ch["highlight_path"]:
                            st.image(ch["highlight_path"])

                # Save CSV Report
                csv_path = OUTPUT_DIR / "report.csv"
                generate_report(real_changes, str(csv_path))

                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Audit Report (CSV)",
                        data=f,
                        file_name="docsentinel_report.csv",
                        mime="text/csv"
                    )
