

from __future__ import annotations

from ai_tutor.data_utils import prepare_all_splits


def main() -> None:
    print("=== Download and Prepare Data ===")
    prepare_all_splits()
    print("Data preparation completed.")


if __name__ == "__main__":
    main()
