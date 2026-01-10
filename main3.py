# main2.py
from docsentinel2.visual_diff import run_visual_diff

def main():
    old_path = "data/v1.pdf"
    new_path = "data/v2.pdf"

    print("Running visual diff...")
    changes = run_visual_diff(old_path, new_path)

    if not changes:
        print("No visual changes detected ✅")
    else:
        for i, ch in enumerate(changes, 1):
            print(f"=== Visual Change #{i} ===")
            for k, v in ch.items():
                print(f"{k}: {v}")
            print()

if __name__ == "__main__":
    main()
