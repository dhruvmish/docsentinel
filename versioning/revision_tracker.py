# versioning/revision_tracker.py

import json
import os
from docsentinel2.visual_diff import run_visual_diff
from docsentinel2.diff_engine_onnx import DiffEngineONNX
from docsentinel2.table_diff import diff_workbooks
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.diff_engine_onnx import DiffEngineONNX

class RevisionTracker:

    def __init__(self, store):
        self.store = store

        # --- REQUIRED DEPENDENCIES ---
        self.embedder = EmbeddingModel()
        self.aligner = SemanticAligner(self.embedder)

        # --- TEXT DIFF ENGINE ---
        self.text_engine = DiffEngineONNX(
            embedder=self.embedder,
            aligner=self.aligner
        )

    def diff_versions(
        self,
        doc_id,
        branch,
        old_version,
        new_version
    ):
        old_path = self.store.get_version_path(doc_id, branch, old_version)
        new_path = self.store.get_version_path(doc_id, branch, new_version)

        changes = []

        # Text
        changes.extend(
            self.text_engine.detect_changes(old_path, new_path)
        )

        # Visual
        changes.extend(
            run_visual_diff(old_path, new_path)
        )

        # Tables (if needed)
        try:
            changes.extend(
                diff_workbooks(old_path, new_path)
            )
        except Exception:
            pass

        # Persist changes
        out_path = os.path.join(
            os.path.dirname(new_path),
            f"changes_from_{old_version}.json"
        )
        with open(out_path, "w") as f:
            json.dump(changes, f, indent=2)

        return changes
