import pandas as pd
import re


def extract_two_numbers(text):
    if pd.isna(text):
        return pd.Series([0.0, 0.0])

    text = str(text)

    # Match German or English decimals: 3,5 or 3.5 or 3
    nums = re.findall(r"\d+[.,]?\d*", text)

    # Convert to float, replace comma with dot
    nums = [float(n.replace(",", ".")) for n in nums]

    num1 = nums[0] if len(nums) > 0 else 0.0
    num2 = nums[1] if len(nums) > 1 else 0.0

    return pd.Series([num1, num2])


def extract_unit(text):
    if pd.isna(text):
        return ""
    m = re.search(r"(mm|cm)", str(text).lower())
    return m.group(1) if m else ""

def extract_size(df, num1, num2, size):
    # Extract numbers
    df[[num1, num2]] = df[size].apply(extract_two_numbers)

    # Extract unit
    df[num1 + "_unit"] = df[size].apply(extract_unit)
    df[num1 + "_mm"] = df[num1]
    df[num2 + "_mm"] = df[num2]

    # Multiply by 10 if unit is cm
    mask = df[num1 + "_unit"] == "cm"
    df.loc[mask, num1 + "_mm"] = df.loc[mask, num1 + "_mm"] * 10
    df.loc[mask, num2 + "_mm"] = df.loc[mask, num2 + "_mm"] * 10

    # Swap if num2_mm > num1_mm
    mask = df[num2 + "_mm"] > df[num1 + "_mm"]
    df.loc[mask, [num1 + "_mm", num2 + "_mm"]] = (
        df.loc[mask, [num2 + "_mm", num1 + "_mm"]].values
    )
    return df

def move_columns_after(df, cols_to_move, after_col):
    cols = list(df.columns)

    for c in cols_to_move:
        cols.remove(c)

    insert_at = cols.index(after_col) + 1
    new_cols = cols[:insert_at] + cols_to_move + cols[insert_at:]

    return df[new_cols]


df = pd.read_excel("llm_out_stage2.xlsx")

# Reset index to 0..n-1
df = df.reset_index(drop=True)
df["index"] = df.index
df["answer"] = df["answer"].astype(str).str.replace(r"[\[\]]", "", regex=True)
df["answer_raw"] = df["answer"]

df = df.assign(answer=df["answer"].str.split("\n")).explode("answer")

df["has_semicolon"] = df["answer"].str.contains(";", na=False)

# Fallback for rows without semicolon → ensures split always works
df["answer_for_split"] = df["answer"].where(df["has_semicolon"], "; ; ")

df[["location_out", "size_out", "en_bloc_out"]] = (
    df["answer_for_split"]
      .str.split(";", n=2, expand=True)
      .apply(lambda col: col.str.strip())
)


df["location_out"] = df["location_out"].str.replace(
    r"^\s*-+\s*(?=[A-Za-z0-9])", "", regex=True
)

# Extract size_out
df = extract_size(df, "num1", "num2", "size_out")

# Extract Durchmesser
df = extract_size(df, "Durchm1", "Durchm2", "Durchmesser")

df = move_columns_after(
    df,
    ["Durchm1", "Durchm2", "Durchm1_unit", "Durchm1_mm", "Durchm2_mm"],
    "Durchmesser"
)

# Find max row
df["max"] = (
    df["num1_mm"] == df.groupby("index")["num1_mm"].transform("max")
).map({True: 1, False: 0})

# Mark rows with two numbers
df["two_num"] = ((df["num1_mm"] != 0) & (df["num2_mm"] != 0)).astype(int)

df["multiple_polyps"] = pd.Series(
    df.index.duplicated(keep=False), index=df.index
).map({True: "ja", False: "nein"})

# Keep only row with largest size
df = (
    df[df["max"] == 1]
      .sort_values(
          ["index", "num1_mm", "num2_mm", "two_num"],
          ascending=[True, False, False, False]
      )
      .groupby("index", as_index=False)
      .first()
)
# --- Add comparison columns ---

# Compare Durchmesser vs. size_out
df["match_Durchm1"] = (df["Durchm1_mm"] == df["num1_mm"]).astype(int)
df["match_Durchm2"] = (df["Durchm2_mm"] == df["num2_mm"]).astype(int)

# Compare Abtragungsart vs. en_bloc_out
df["match_Abtragungsart"] = (df["Abtragungsstatus"] == df["en_bloc_out"]).astype(int)

# Compare mehrere_Polypen vs. multiple_polyps
df["match_mehrere_Polypen"] = (df["mehrere_Polypen"] == df["multiple_polyps"]).astype(int)
df["location_out"] = df["location_out"].astype(str).str.replace(r"\bKolon\b", "Colon", regex=True, flags=re.IGNORECASE)

df.to_excel("llm_out_stage3.xlsx", index=False)

