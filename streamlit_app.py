# streamlit_app.py


import os
import tempfile
from pathlib import Path
from typing import List, Dict


import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageDraw


from docsentinel2.ingestion import load_document, load_xlsx
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.visual_diff import run_visual_diff
from docsentinel2.table_diff import diff_workbooks
from docsentinel2.summarizer import summarize_change






# Semantic engine (ONNX if available)
try:
   from docsentinel2.diff_engine_onnx import DiffEngineONNX as SemanticEngine
   ONNX_AVAILABLE = True
except ImportError:
   from docsentinel2.diff_engine import DiffEngine as SemanticEngine
   ONNX_AVAILABLE = False


# Simple diff engine
from docsentinel2.diff_engine_simple import DiffEngineSimple




# ---------------------------------------------------------
# Helper: PDF rendering + scale from page coords → pixels
# ---------------------------------------------------------
def render_pdf_page(pdf_path: str, page_num: int):
   """
   Render 1-based page number of a PDF to a PIL image.
   Returns (image, (scale_x, scale_y)) so we can map bbox from page coords.
   """
   doc = fitz.open(pdf_path)
   page = doc.load_page(page_num - 1)
   pix = page.get_pixmap()
   img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


   rect = page.rect
   scale_x = pix.width / rect.width
   scale_y = pix.height / rect.height
   doc.close()
   return img, (scale_x, scale_y)




def draw_bbox(img: Image.Image, bbox, scale, color="red", width=4):
   """
   Draw a rectangle defined in page coordinates onto the rendered image.
   bbox is (x0, y0, x1, y1) in PDF page coords.
   """
   if not bbox:
       return img
   sx, sy = scale
   x0, y0, x1, y1 = bbox
   out = img.copy()
   draw = ImageDraw.Draw(out)
   draw.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                  outline=color, width=width)
   return out




# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------


st.set_page_config(page_title="DocSentinel", layout="wide")
st.title("DocSentinel – AI Document Change Auditor")


with st.sidebar:
   st.header("⚙️ Comparison Mode")
   mode = st.radio(
       "Text comparison mode:",
       ["Semantic (AI)", "Simple (Diff)"],
       index=0,
   )
   filter_minor = st.checkbox("Hide Minor Text Changes", True)
   show_raw = st.checkbox("Show Raw JSON", False)
   export_split = st.checkbox(
       "Export audit-grade split reports (Text / Image / Table)"
   )
   st.caption(
       "Semantic (AI): SBERT + RoBERTa NLI\n"
       "Simple (Diff): exact text blocks only\n"
       "Visual & Tables are unchanged in both modes."
   )


col1, col2 = st.columns(2)
with col1:
   old_file = st.file_uploader("Old Document", type=["pdf", "xlsx"])
with col2:
   new_file = st.file_uploader("New Document", type=["pdf", "xlsx"])


if st.button("🔍 Compare Documents", use_container_width=True):
   if not old_file or not new_file:
       st.error("Upload both documents.")
       st.stop()


   ext = Path(old_file.name).suffix.lower()
   if ext not in (".pdf", ".xlsx"):
       st.error("Only PDF and XLSX are supported.")
       st.stop()


   def save_tmp(f):
       t = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
       t.write(f.getbuffer())
       t.close()
       return t.name


   old_path = save_tmp(old_file)
   new_path = save_tmp(new_file)


   with st.spinner(f"Running DocSentinel ({mode})..."):
       changes: List[Dict] = []


       if ext == ".pdf":
           # ------------ TEXT DIFF -------------
           if mode == "Simple (Diff)":
               # Use our new simple engine (block-based, exact text)
               eng = DiffEngineSimple()
               text_changes = eng.detect_changes(
                   old_path=old_path,
                   new_path=new_path,
               )
           else:
               # Semantic AI mode
               old_sents = load_document(old_path)
               new_sents = load_document(new_path)
               embed = EmbeddingModel()
               align = SemanticAligner(embed)
               try:
                   eng = SemanticEngine(
                       embedder=embed,
                       aligner=align,
                       old_path=old_path,
                       new_path=new_path,
                   )
               except TypeError:
                   eng = SemanticEngine(embedder=embed, aligner=align)
               text_changes = eng.detect_changes(old_sents, new_sents)


           changes.extend(text_changes)


           # ------------ VISUAL DIFF -----------
           changes.extend(run_visual_diff(old_path, new_path))


       else:
           # XLSX: tables only
           wb_changes = diff_workbooks(
               load_xlsx(old_path),
               load_xlsx(new_path),
               EmbeddingModel(),
           )
           changes.extend(wb_changes)


   # --------------------------------------------------
   # Filter out NO_CHANGE / MINOR_CHANGE if requested
   # --------------------------------------------------
   filtered = [
       c for c in changes
       if c.get("label") != "NO_CHANGE"
       and not (filter_minor and c.get("label") == "MINOR_CHANGE")
   ]


   if not filtered:
       st.success("🎉 No meaningful changes detected.")
       st.stop()


   # Categorize
   text_changes = [c for c in filtered if "old" in c and "new" in c]


   VISUAL_TYPES = {
       "IMAGE_REPLACED",
       "IMAGE_CHANGE_REGION",
       "IMAGE_TEXT_CHANGED",
       "PAGE_LAYOUT_CHANGED",
       "IMAGE_ADDED",
       "IMAGE_REMOVED",
       "IMAGE_SHIFTED",
       "IMAGE_MINOR_TWEAK",
   }
   visual_changes = [c for c in filtered if c.get("type") in VISUAL_TYPES]
   table_changes = [c for c in filtered if c.get("sheet") or c.get("sheet_name")]


   st.subheader("📊 Summary")
   st.write(f"📝 Text changes: **{len(text_changes)}**")
   st.write(f"🖼 Visual changes: **{len(visual_changes)}**")
   st.write(f"📊 Table changes: **{len(table_changes)}**")


   tab_text, tab_visual, tab_tables = st.tabs(["📝 Text", "🖼 Visual", "📊 Tables"])


   # --------------------------------------------------
   # TEXT TAB (both modes, same viewer)
   # --------------------------------------------------
   with tab_text:
       if not text_changes:
           st.info("No text changes detected.")
       else:
           for i, c in enumerate(text_changes, 1):
               label = c.get("label", "CHANGE")
               with st.expander(f"#{i} — {label}", expanded=False):
                   # Summary: semantic mode uses full summarizer;
                   # simple mode: basic description.
                   if mode == "Semantic (AI)":
                       st.markdown(f"**Summary:** {summarize_change(c)}")
                   else:
                       label = c.get("label")
                       if label == "TEXT_ADDED":
                           st.markdown("➕ **Text Added**")
                       elif label == "TEXT_REMOVED":
                           st.markdown("➖ **Text Removed**")
                       elif label == "TEXT_MODIFIED":
                           st.markdown("✍️ **Text Modified**")
                       else:
                           st.markdown("🔄 **Text Changed**")


                   c1, c2 = st.columns(2)


                   # Old side
                   page_old = c.get("page_old")
                   bbox_old = c.get("bbox_old")
                   if page_old:
                       try:
                           img_old, scale_old = render_pdf_page(old_path, page_old)
                           img_old = draw_bbox(img_old, bbox_old, scale_old, "red")
                           c1.image(
                               img_old,
                               caption=f"Old (Page {page_old})",
                               use_column_width=True,
                           )
                       except Exception as e:
                           c1.warning(f"Old preview error: {e}")
                   else:
                       c1.info("No old page information.")


                   # New side
                   page_new = c.get("page_new")
                   bbox_new = c.get("bbox_new")
                   if page_new:
                       try:
                           img_new, scale_new = render_pdf_page(new_path, page_new)
                           img_new = draw_bbox(img_new, bbox_new, scale_new, "green")
                           c2.image(
                               img_new,
                               caption=f"New (Page {page_new})",
                               use_column_width=True,
                           )
                       except Exception as e:
                           c2.warning(f"New preview error: {e}")
                   else:
                       c2.info("No new page information.")


                   st.write("**Old text:**", c.get("old", ""))
                   st.write("**New text:**", c.get("new", ""))


                   if show_raw:
                       st.json(c)


   # --------------------------------------------------
   # VISUAL TAB
   # --------------------------------------------------
   with tab_visual:
       if not visual_changes:
           st.info("No visual changes detected.")
       else:
           for i, v in enumerate(visual_changes, 1):
               with st.expander(f"#{i} — {v.get('type')}", expanded=False):
                   st.markdown(f"**Summary:** {summarize_change(v)}")
                   page = v.get("page")
                   regions = v.get("regions") or []


                   if page:
                       try:
                           img_old, s_old = render_pdf_page(old_path, page)
                           img_new, s_new = render_pdf_page(new_path, page)


                           # 🔴🟢 FULL IMAGE BOUNDING BOX (like text diff)
                           bbox_old = v.get("bbox_old")
                           bbox_new = v.get("bbox_new")


                           if bbox_old:
                               img_old = draw_bbox(img_old, bbox_old, s_old, "red", width=4)


                           if bbox_new:
                               img_new = draw_bbox(img_new, bbox_new, s_new, "green", width=4)


                           # Existing region-level boxes (DO NOT TOUCH)
                           for b in regions:
                               img_old = draw_bbox(img_old, b, s_old, "red", width=2)
                               img_new = draw_bbox(img_new, b, s_new, "green", width=2)


                           c1, c2 = st.columns(2)
                           c1.image(img_old, caption=f"Old (Page {page})", use_column_width=True)
                           c2.image(img_new, caption=f"New (Page {page})", use_column_width=True)
                       except Exception as e:
                           st.warning(f"Visual preview error: {e}")


                   if v.get("highlight_path") and os.path.exists(v["highlight_path"]):
                       st.image(v["highlight_path"], caption="Diff heatmap", use_column_width=True)


                   if show_raw:
                       st.json(v)


   # --------------------------------------------------
   # TABLE TAB
   # --------------------------------------------------
   with tab_tables:
       if not table_changes:
           st.info("No table changes detected.")
       else:
           for i, t in enumerate(table_changes, 1):
               with st.expander(f"#{i}", expanded=False):
                   st.write(summarize_change(t))
                   if show_raw:
                       st.json(t)


   # --------------------------------------------------
   # CSV Export
   # --------------------------------------------------
   from docsentinel2.report import generate_report, generate_audit_reports


   audit_dir = None
   if export_split:
       audit_dir = os.path.join(tempfile.gettempdir(), "ds_audit_reports")
       generate_audit_reports(filtered, audit_dir)


   import shutil


   # ---- Single CSV (unchanged) ----
   csv_path = os.path.join(tempfile.gettempdir(), "ds_report.csv")
   generate_report(filtered, csv_path)


   with open(csv_path, "rb") as f:
       st.download_button(
           "⬇ Download Consolidated CSV",
           f,
           "docsentinel_report.csv"
       )


   # ---- Split audit reports (ZIP) ----
   if export_split and audit_dir and os.path.exists(audit_dir):
       zip_path = shutil.make_archive(
           audit_dir,
           "zip",
           audit_dir
       )
       with open(zip_path, "rb") as f:
           st.download_button(
               "⬇ Download Audit Reports (ZIP)",
               f,
               "docsentinel_audit_reports.zip"
           )
   st.write("DEBUG image bbox:", bbox_old)
   st.success("✔ Audit complete.")

