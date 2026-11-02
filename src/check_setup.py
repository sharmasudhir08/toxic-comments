"""Quick sanity check:  python -m src.check_setup"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS
from src.data import load_official_test, load_train


def main(data_dir="data"):
    print("Python:", sys.version.split()[0])
    try:
        import torch, transformers
        print("torch:", torch.__version__, "| transformers:", transformers.__version__)
        if torch.cuda.is_available():
            print("GPU  :", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("GPU  : Apple MPS")
        else:
            print("GPU  : none (CPU only) -> use --max_train_samples for faster runs")
    except ImportError as e:
        print("Missing package:", e, "-> run: pip install -r requirements.txt")
        return

    df = load_train(data_dir)
    print(f"\ntrain.csv: {len(df)} rows")
    print(df[LABELS].sum().to_string())
    test = load_official_test(data_dir)
    if test is None:
        print("\ntest.csv / test_labels.csv not found -> a 10% hold-out of train.csv will be used as test set.")
    else:
        print(f"\nofficial test set: {len(test)} scored rows (the paper used 63,978)")
    print("\nSetup looks good.")


if __name__ == "__main__":
    main()
