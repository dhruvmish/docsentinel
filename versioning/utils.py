def next_patch_version(base_version: str) -> str:
    major, minor = base_version.split(".")
    return f"{major}.{int(minor) + 1}"
