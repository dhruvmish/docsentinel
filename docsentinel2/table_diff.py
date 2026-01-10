# docsentinel2/table_diff.py

from typing import List, Dict, Any
from .ingestion import WorkbookRep, TableRep
from difflib import SequenceMatcher


def _align_indices(old_vals: List[Any], new_vals: List[Any]) -> List[tuple]:
    """
    Align rows or columns based on SequenceMatcher
    Returns list of index pairs: [(old_idx, new_idx), ...]
    """
    matcher = SequenceMatcher(None, old_vals, new_vals)
    mapping = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for a, b in zip(range(i1, i2), range(j1, j2)):
                mapping.append((a, b))
        # mismatched regions handled later as added/removed
    return mapping


def _is_number(value):
    try:
        float(value)
        return True
    except:
        return False


def diff_workbooks(old: WorkbookRep, new: WorkbookRep, embedder=None,
                   num_threshold: float = 0.01):
    """
    Compare tables sheet-wise and cell-wise.
    embedder: SentenceEmbedder instance for semantic text comparison.
    num_threshold: % difference threshold to count as numeric change.
    """
    changes = []

    old_sheets = {t.sheet_name: t for t in old.tables}
    new_sheets = {t.sheet_name: t for t in new.tables}

    # ---------- Detect removed sheets ----------
    for sname in old_sheets:
        if sname not in new_sheets:
            changes.append({"type": "SHEET_REMOVED", "sheet": sname})

    # ---------- Detect added sheets ----------
    for sname in new_sheets:
        if sname not in old_sheets:
            changes.append({"type": "SHEET_ADDED", "sheet": sname})

    # ---------- Compare common sheets ----------
    for sname in set(old_sheets).intersection(new_sheets):
        old_t = old_sheets[sname]
        new_t = new_sheets[sname]

        # Build matrices for easier indexing
        def build_matrix(t: TableRep):
            matrix = [["" for _ in range(t.max_col)] for _ in range(t.max_row)]
            for cell in t.cells:
                r = cell.row - 1
                c = cell.col - 1
                if r < t.max_row and c < t.max_col:
                    matrix[r][c] = str(cell.value)
            return matrix

        M_old = build_matrix(old_t)
        M_new = build_matrix(new_t)

        row_map = _align_indices(list(range(old_t.max_row)), list(range(new_t.max_row)))
        col_map = _align_indices(list(range(old_t.max_col)), list(range(new_t.max_col)))

        old_rows = set(range(old_t.max_row))
        new_rows = set(range(new_t.max_row))
        old_cols = set(range(old_t.max_col))
        new_cols = set(range(new_t.max_col))

        aligned_old_rows = {o for o, _ in row_map}
        aligned_new_rows = {n for _, n in row_map}
        aligned_old_cols = {o for o, _ in col_map}
        aligned_new_cols = {n for _, n in col_map}

        removed_rows = sorted(list(old_rows - aligned_old_rows))
        added_rows = sorted(list(new_rows - aligned_new_rows))
        removed_cols = sorted(list(old_cols - aligned_old_cols))
        added_cols = sorted(list(new_cols - aligned_new_cols))

        for r in removed_rows:
            changes.append({"type": "ROW_REMOVED", "sheet": sname, "row": r+1})

        for r in added_rows:
            changes.append({"type": "ROW_ADDED", "sheet": sname, "row": r+1})

        for c in removed_cols:
            changes.append({"type": "COLUMN_REMOVED", "sheet": sname, "col": c+1})

        for c in added_cols:
            changes.append({"type": "COLUMN_ADDED", "sheet": sname, "col": c+1})

        # ---------- Compare aligned rows/cols ----------
        for (orow, nrow) in row_map:
            for (ocol, ncol) in col_map:
                old_val = M_old[orow][ocol]
                new_val = M_new[nrow][ncol]

                if old_val == new_val:
                    continue

                change = {
                    "type": None,
                    "sheet": sname,
                    "row": nrow + 1,
                    "col": ncol + 1,
                    "old": old_val,
                    "new": new_val
                }

                # Numeric diff
                if _is_number(old_val) and _is_number(new_val):
                    num_old = float(old_val)
                    num_new = float(new_val)
                    diff_pct = abs(num_new - num_old) / max(abs(num_old), 1e-9)

                    if diff_pct >= num_threshold:
                        change["type"] = "NUMERIC_CELL_CHANGED"
                        change["percent_change"] = round(diff_pct * 100, 2)

                else:
                    # Semantic diff using embedder
                    if embedder:
                        sim = embedder.similarity(old_val, new_val)
                        change["cosine"] = sim
                        change["type"] = "SEMANTIC_CELL_CHANGED" if sim < 0.95 else "TEXT_MINOR_UPDATE"
                    else:
                        change["type"] = "TEXT_CHANGED"

                if change["type"]:
                    changes.append(change)

    return changes
