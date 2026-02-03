# DocSentinel

**An AI-Powered, Multi-Modal Document Revision Tracking System**

## 1. Problem Statement

Organizations rely on documents whose meaning, structure, and visual layout must remain consistent across revisions.

However, traditional document comparison tools suffer from fundamental limitations:

### Limitations of Existing Systems

1. **Lexical-only comparison**
   - Line-by-line or token-based diffs
   - Cannot detect paraphrasing or semantic drift

2. **No spatial grounding**
   - Changes are not mapped back to exact page regions
   - Makes auditing and review difficult

3. **Visual blindness**
   - Image replacements, layout shifts, and diagram changes are ignored

4. **Poor explainability**
   - No structured output
   - No confidence or classification of change severity

### Core Challenge

To build a system that can:
- Detect what changed
- Understand how it changed (semantic vs superficial)
- Localize where it changed (page + region)
- Work reliably across heterogeneous document content

## 2. System Design Philosophy

DocSentinel is built around four principles:

1. Semantic-first comparison
2. Spatially grounded outputs
3. Multi-modal detection (text + vision + tables)
4. Stability over experimental complexity

Every pipeline component adheres to these constraints.

## 3. Technology Stack

### Core Libraries

| Purpose | Library |
|---------|---------|
| UI | streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Image processing | PIL, OpenCV |
| NLP embeddings | sentence-transformers (SBERT) |
| Similarity | numpy, scipy |
| Visual similarity | imagehash, skimage.metrics |
| OCR | pytesseract |
| ML inference | onnxruntime, torch |
| Excel diff | pandas, openpyxl |

## 4. Text Difference Pipeline (PDF)

DocSentinel supports two text diff modes that share the same output schema.

### 4.1 PDF Text Ingestion & Spatial Extraction

**Technique Used**
- PyMuPDF (fitz) page parsing
- Sentence-level extraction
- Bounding box capture for each sentence

**Why Sentence-Level?**
- Paragraphs are too coarse
- Tokens are too fine
- Sentences preserve semantic units while remaining spatially localizable

**Extracted Data Structure**

```json
{
  "text": "The drug dosage was increased to 50 mg.",
  "page": 4,
  "bbox": [x0, y0, x1, y1]
}
```

**Coordinate System**
- PDF-native coordinate system (origin bottom-left)
- Converted later to pixel coordinates during rendering

### 4.2 Semantic (AI-Based) Text Diff Mode

**Overview**

This mode detects meaningful changes, even when wording differs.

**Step 1: Sentence Embeddings**

*Model Used*
- SBERT (Sentence-BERT)

*Why SBERT?*
- Produces semantically meaningful embeddings
- Optimized for cosine similarity
- Robust to paraphrasing

*Mathematics:*

Each sentence is mapped to a vector:

**Vector Definition:** eᵢ ∈ ℝᵈ

Similarity between old and new sentences:

**Cosine Similarity:** cosine_sim(a, b) = (a ⋅ b) / (‖a‖ ‖b‖)

**Step 2: Semantic Alignment**

*Goal*

Match sentences from old and new documents.

*Technique*
- Pairwise cosine similarity
- Greedy or threshold-based matching

*Threshold Logic*
- High similarity → unchanged or minor change
- Medium similarity → modified
- No match → added or removed

**Step 3: Change Classification (NLI-style Logic)**

Rather than using a heavy NLI model, DocSentinel applies rule-based semantic reasoning:

| Condition | Classification |
|-----------|----------------|
| No old match | TEXT_ADDED |
| No new match | TEXT_REMOVED |
| Moderate similarity + lexical drift | TEXT_MODIFIED |
| High similarity | MINOR_CHANGE |

**Step 4: Output Schema (Unified)**

```json
{
  "label": "TEXT_MODIFIED",
  "old": "...",
  "new": "...",
  "page_old": 2,
  "bbox_old": [...],
  "page_new": 2,
  "bbox_new": [...]
}
```

This schema is contractually enforced across pipelines.

### 4.3 Simple (Rule-Based) Text Diff Mode

**Motivation**
- Faster
- No embedding models
- Acts as a baseline

**Critical Design Choice**

This mode does NOT reuse semantic sentence extraction.

*Why?*
- Avoids misaligned bounding boxes
- Prevents duplicate diffs
- Ensures UI consistency

**Technique Used**

1. Independent PDF text extraction
2. Page-aware string diff
3. Heuristic merging of old/new changes

**String Diff Logic**
- Python difflib-style comparison
- Context-aware grouping
- Merged into unified changes

**Mathematical Nature**

This mode is deterministic, relying on:
- Longest Common Subsequence (LCS)
- Edit distance heuristics

## 5. Visual Difference Pipeline (PDF Images)

Text-based comparison fails for:
- Diagrams
- Charts
- Scanned documents
- Layout-only changes

### 5.1 Page Alignment

Ensures:
- Page-to-page correspondence
- Consistent coordinate space

### 5.2 Perceptual Hashing (pHash)

**Library**
- imagehash

**Technique**
- Converts image to frequency domain
- Generates compact hash

**Distance Metric**

Hamming Distance(h₁, h₂)

Used to classify:
- Strong vs weak visual changes

### 5.3 OCR + Semantic Validation

**OCR**
- pytesseract

**Purpose**

Distinguish between:
- Image text changes
- Visual-only changes

**Post-OCR**
- Extracted text is embedded (SBERT)
- Semantic similarity used to validate OCR noise

### 5.4 Siamese CNN Feature Similarity

**Technique**
- CNN feature embeddings from image regions
- Siamese-style similarity comparison

**Similarity Score**

sim = 1 - ‖f(x₁) - f(x₂)‖

Used for:
- Region-level change confidence

### 5.5 SSIM (Structural Similarity Index)

**Library**
- skimage.metrics.structural_similarity

**Formula**

SSIM(x,y) = ((2μₓμᵧ + C₁)(2σₓᵧ + C₂)) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))

Used to detect:
- Layout shifts
- Page-level structural changes

### 5.6 Visual Change Outputs

```json
{
  "type": "IMAGE_CHANGE_REGION",
  "page": 3,
  "regions": [[x0, y0, x1, y1]],
  "highlight_path": "...",
  "siamese_similarity": 0.91
}
```

## 6. Table Difference Pipeline (Excel)

**Techniques Used**
- pandas DataFrame comparison
- Index alignment
- Cell-wise diffing

**Change Types**
- Row added / removed
- Cell value modified
- Structural differences

## 7. Streamlit UI Architecture

**Rendering Logic**
- Page rendered using PyMuPDF pixmaps
- PDF coordinates → pixel scaling:

x_pixel = x_pdf × (W_img / W_page)

**Stability Measures**
- Unified schemas
- Defensive null checks
- Lazy rendering
- No model inference in UI thread

## 8. Performance Evaluation Methodology

### 8.1 Text Diff Evaluation

**Metrics**
- Precision / Recall (sentence-level)
- Semantic false positives
- Missed semantic changes

**Method**
- Manually curated document pairs
- Injected paraphrases
- Controlled wording variations

### 8.2 Visual Diff Evaluation

**Metrics**
- Region detection accuracy
- False highlighting rate
- Missed image changes

**Method**
- Visual overlay inspection
- Hash distance vs SSIM correlation

## 9. Engineering Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Avoid deep image segmentation | Stability |
| Unified schemas | UI safety |
| ONNX inference | Speed & portability |
| Sentence-level diff | Best semantic granularity |

## 10. Conclusion

DocSentinel is a system-level solution, not a single-model demo.

It integrates NLP, computer vision, and document engineering to produce auditable, explainable, and spatially grounded document diffs.
