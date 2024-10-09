from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    print(project_dir)
