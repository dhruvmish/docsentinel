from docsentinel2.visual_diff import detect_visual_changes

old_doc_path = "data/v1.pdf"
new_doc_path = "data/v2.pdf"

print("\nRunning visual diff...")
visual_results = detect_visual_changes(old_doc_path, new_doc_path)

for i, ch in enumerate(visual_results, 1):
    print(f"=== Visual Change #{i} ===")
    print(f"{ch}")
