import pandas as pd
import os
import re


def parse_triglavian_buy_orders(txt_path, output_csv="util/triglavian_buy_stations.csv"):
    if not os.path.exists(txt_path):
        print(f"Error: file '{txt_path}' not found.")
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Clean and split into rows
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    rows = []
    for line in lines:
        # Prefer splitting by tab if present, fallback to regex
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        else:
            normalized = re.sub(r"[ ]{2,}", ",", line)
            parts = [p.strip() for p in normalized.split(",") if p.strip()]

        if len(parts) >= 6:
            rows.append(parts[:6])

    if not rows:
        print("No valid rows found.")
        return None

    df = pd.DataFrame(rows, columns=[
        "region", "system_name", "station_name", "sec_status",
        "price", "volume_remain"
    ])

    # Clean types
    df["sec_status"] = pd.to_numeric(df["sec_status"].str.replace(",", "."), errors="coerce")
    df["price"] = pd.to_numeric(df["price"].str.replace(".", "", regex=False), errors="coerce")
    df["volume_remain"] = pd.to_numeric(df["volume_remain"].str.replace(".", "", regex=False), errors="coerce").fillna(0).astype(int)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    df.to_csv(os.path.join(os.path.dirname(txt_path), "triglavian_buy_orders_cleaned.csv"), index=False)
    print(f"Parsed {len(df)} entries and saved to '{output_csv}' and '{os.path.dirname(txt_path)}/triglavian_buy_orders_cleaned.csv'.")
    return df


# Example usage:
df = parse_triglavian_buy_orders(r"Trig_database_buyer_list\trig_list.txt")
