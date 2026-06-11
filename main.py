from pathlib import Path

from src.data_preprocessing import save_clean_data
from src.utils import RAW_DATA, PROCESSED_DATA


def main() -> None:
    print("Running data preprocessing pipeline...")
    clean_df = save_clean_data(RAW_DATA, PROCESSED_DATA)
    print(f"Cleaned data saved at: {PROCESSED_DATA}")
    print(f"Rows: {clean_df.shape[0]}, Columns: {clean_df.shape[1]}")


if __name__ == "__main__":
    main()
