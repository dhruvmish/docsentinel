import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List


class VersionStore:
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Document & branch initialization
    # --------------------------------------------------
    def init_document(self, doc_id: str):
        doc_path = self.base_dir / doc_id
        doc_path.mkdir(exist_ok=True)

        registry = doc_path / "registry.json"
        if not registry.exists():
            registry.write_text(json.dumps({
                "doc_id": doc_id,
                "branches": {}
            }, indent=2))

    def create_branch(self, doc_id: str, branch: str, parent_branch: Optional[str]):
        registry = self._load_registry(doc_id)

        if branch in registry["branches"]:
            return

        registry["branches"][branch] = {
            "parent_branch": parent_branch,
            "versions": []
        }

        self._save_registry(doc_id, registry)

        branch_path = self.base_dir / doc_id / branch
        branch_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 🔑 VERSION ID GENERATION (THIS FIXES YOUR ERROR)
    # --------------------------------------------------
    def next_version_id(self, doc_id: str, branch: str) -> str:
        registry = self._load_registry(doc_id)
        versions = registry["branches"][branch]["versions"]

        if not versions:
            return "v1"

        last = versions[-1]  # e.g. v1, v2, v2.1
        parts = last.lstrip("v").split(".")

        if len(parts) == 1:
            return f"v{int(parts[0]) + 1}"
        else:
            major = parts[0]
            minor = int(parts[1]) + 1
            return f"v{major}.{minor}"

    # --------------------------------------------------
    # Version persistence
    # --------------------------------------------------
    def add_version(
        self,
        doc_id: str,
        branch: str,
        version_id: str,
        parent_version: Optional[str],
        file_path: str
    ):
        registry = self._load_registry(doc_id)

        version_dir = self.base_dir / doc_id / branch / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy document
        target = version_dir / "document.pdf"
        shutil.copy(file_path, target)

        registry["branches"][branch]["versions"].append(version_id)

        registry.setdefault("versions", {})
        registry["versions"][version_id] = {
            "branch": branch,
            "parent": parent_version
        }

        self._save_registry(doc_id, registry)

    # --------------------------------------------------
    # Query helpers
    # --------------------------------------------------
    def get_branch_head(self, doc_id: str, branch: str) -> Optional[str]:
        registry = self._load_registry(doc_id)
        versions = registry["branches"][branch]["versions"]
        return versions[-1] if versions else None

    def get_version_path(self, doc_id: str, branch: str, version_id: str) -> str:
        return str(self.base_dir / doc_id / branch / version_id / "document.pdf")

    # --------------------------------------------------
    # Registry helpers
    # --------------------------------------------------
    def _load_registry(self, doc_id: str) -> Dict:
        registry_path = self.base_dir / doc_id / "registry.json"
        return json.loads(registry_path.read_text())

    def _save_registry(self, doc_id: str, data: Dict):
        registry_path = self.base_dir / doc_id / "registry.json"
        registry_path.write_text(json.dumps(data, indent=2))
