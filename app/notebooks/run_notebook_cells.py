import subprocess
import sys
import traceback

import joblib
import nbformat


def main():
    # Execute notebook code cells one-by-one in shared state.
    nb_path = r"phase1/notebooks/phase1_v3_colab.ipynb"
    local_data = r"dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv"

    nb = nbformat.read(nb_path, as_version=4)
    globals_state = {"joblib": joblib}
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    total = len(code_cells)
    print(f"Executing {total} code cells...")

    for idx, cell in enumerate(code_cells, 1):
        source = cell.source

        # Replace colab default path with local workspace path at runtime.
        if "DATA_PATH = '/content/final_labeled_fake_reviews.csv'" in source:
            source = source.replace(
                "DATA_PATH = '/content/final_labeled_fake_reviews.csv'",
                f"DATA_PATH = r'{local_data}'",
            )

        # Handle the pip cell written as notebook shell command.
        if "!pip -q install" in source:
            print(f"Cell {idx}/{total}: installing dependencies...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "nltk",
                    "scipy",
                    "joblib",
                ],
                check=True,
            )
            print(f"Cell {idx}/{total}: OK")
            continue

        print(f"Cell {idx}/{total}: running...")
        try:
            exec(compile(source, f"<cell_{idx}>", "exec"), globals_state)
            print(f"Cell {idx}/{total}: OK")
        except Exception:
            print(f"Cell {idx}/{total}: FAILED")
            traceback.print_exc()
            raise

    print("All notebook code cells executed successfully.")


if __name__ == "__main__":
    main()
