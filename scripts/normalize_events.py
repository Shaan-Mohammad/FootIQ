import argparse
import json
from pathlib import Path

import ijson


def convert_file(input_path: Path, output_dir: Path) -> int:
    """
    Convert one large JSON array into a JSON Lines file.
    Reads one event at a time to reduce memory usage.
    """

    output_path = output_dir / f"{input_path.stem}.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")

    record_count = 0

    try:
        with input_path.open("rb") as source:
            with temporary_path.open("w", encoding="utf-8") as destination:

                # Read each event from the root JSON array
                for event in ijson.items(
                    source, "item", use_float=True
                ):

                    # Write one complete event per line
                    json.dump(
                        event,
                        destination,
                        ensure_ascii=False,
                        separators=(",", ":")
                    )

                    destination.write("\n")
                    record_count += 1

        # Replace the final output only after successful conversion
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return record_count


def main():

    parser = argparse.ArgumentParser(
        description="Convert Wyscout event JSON files into JSON Lines."
    )

    parser.add_argument("input_dir", help="Folder containing original JSON files")
    parser.add_argument("output_dir", help="Folder for converted JSON Lines files")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    input_files = sorted(input_dir.glob("events_*.json"))

    if not input_files:
        raise SystemExit("No events_*.json files found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0

    print("================================")
    print("   FOOTIQ JSON NORMALIZATION")
    print("================================")

    for input_path in input_files:

        print(f"\nProcessing: {input_path.name}")

        count = convert_file(input_path, output_dir)

        print(f"Records converted: {count:,}")

        total_records += count

    print("\n================================")
    print(f"TOTAL RECORDS: {total_records:,}")
    print(f"OUTPUT FOLDER: {output_dir}")
    print("================================")


if __name__ == "__main__":
    main()
