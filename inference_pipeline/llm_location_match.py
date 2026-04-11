import os
import time
import re
import json
from openai import OpenAI
import pandas as pd


# -----------------------------
# Valid target categories
# -----------------------------
VALID_LOCATIONS = [
    "Zökum-linke Flexur",
    "unklar",
    "rechte Flexur",
    "linke Flexur",
    "siehe_Distanz",
    "Zökum",
    "Colon ascendens",
    "Colon transversum",
    "Colon descendens",
    "Sigma",
    "Rektum",
]

VALID_LOCATIONS_LOWER = {v.lower(): v for v in VALID_LOCATIONS}
NAMED_LOCATIONS = [v for v in VALID_LOCATIONS if v not in ("unklar", "siehe_Distanz")]


# -----------------------------
# Helper Functions
# -----------------------------
def extract_first_json_with_location(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-z]*", "", text)
    text = text.replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        if "location" in obj:
            return obj
    except Exception:
        pass
    return None


def normalize_location(raw):
    if not isinstance(raw, str):
        return None
    raw_lower = raw.lower().strip()
    if raw_lower in VALID_LOCATIONS_LOWER:
        return VALID_LOCATIONS_LOWER[raw_lower]
    for key, canonical in VALID_LOCATIONS_LOWER.items():
        if key in raw_lower or raw_lower in key:
            return canonical
    return None


def is_pure_distance(text):
    """Returns True if text contains a number but no named anatomical location."""
    t = str(text)
    has_digit = bool(re.search(r"\d", t))
    has_named = any(loc.lower() in t.lower() for loc in NAMED_LOCATIONS)
    return has_digit and not has_named


def extract_numbers(text):
    """All numbers (incl. decimals) from a text as a set of floats."""
    if not isinstance(text, str):
        return set()
    return set(float(m.replace(",", ".")) for m in re.findall(r"\d+[.,]?\d*", text))


def location_distance_match(lage, distanz, mapped):
    """
    Compares Lage and Distanz_a.a. against the mapped value (location_mapped).
    - Lage == mapped: direct string comparison (normalized)
    - Distanz_a.a. and mapped share the same number: numeric comparison
    Returns (bool, str).
    """
    mapped_str = str(mapped).strip()
    lage_str = str(lage).strip() if not pd.isna(lage) else ""
    dist_str = str(distanz).strip() if not pd.isna(distanz) else ""

    # Lage matches mapped value
    if lage_str.lower() == mapped_str.lower() and lage_str != "":
        return True, f"location_mapped matches Lage ({lage_str})"

    # Matching numbers between Distanz_a.a. and location_mapped
    nums_dist = extract_numbers(dist_str)
    nums_mapped = extract_numbers(mapped_str)
    if nums_dist and nums_mapped and not nums_dist.isdisjoint(nums_mapped):
        return True, f"Matching number(s) in Distanz_a.a. ({dist_str}) and location_mapped ({mapped_str})"

    return False, ""


# -----------------------------
# Configure client
# -----------------------------
client = OpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key=os.environ.get("OLLAMA_API_KEY", "dummy-key"),
)
MODEL = os.environ["OLLAMA_MODEL"]

df = pd.read_excel("llm_out_stage3.xlsx")

VALID_LOCATIONS_STR = "\n".join(f'- "{v}"' for v in VALID_LOCATIONS)
json_suffix = (
    f"\n\nGültige Kategorien:\n{VALID_LOCATIONS_STR}\n\n"
    "Gib ein JSON zurück mit zwei Feldern:\n"
    "1. \"location\": exakt eine der oben genannten Kategorien (exakte Schreibweise)\n"
    "2. \"reason\": kurze Begründung\n"
    "Nur gültiges JSON, nichts anderes."
)

df["location_mapped"] = ""
df["location_mapped_raw"] = ""
df["location_distance_match"] = None
df["location_distance_match_reason"] = ""
failed_rows = []

start_tot = time.time()
processed_counter = 0
SAVE_EVERY = 10
OUTFILE = "llm_out_stage4.xlsx"

for idx, row in df.iterrows():
    start = time.time()
    location_out = row["location_out"]
    lage = row.get("Lage", None)
    distanz = row.get("Distanz_a.a.", None)
    loc_str = str(location_out)

    print(f"\n\nRow: {idx}")
    print("location_out:", location_out)
    print("Lage:", lage, "| Distanz_a.a.:", distanz)
    print("\nInference starts here\n")

    def save_if_due(counter):
        if counter % SAVE_EVERY == 0:
            try:
                df.to_excel(OUTFILE, index=False)
                print(f"Saved progress after {counter} rows.")
            except Exception as e:
                print(f"Error saving file: {e}")

    # 1) Pure distance value: pass through directly
    if is_pure_distance(loc_str):
        final_mapped = loc_str
        print(f"Distance value passed through directly: {loc_str}")

    # 2) Direct normalization mapping
    elif normalize_location(loc_str) is not None:
        final_mapped = normalize_location(loc_str)
        print(f"Direct mapping: {final_mapped}")

    # 3) LLM call
    else:
        prompt = (
            f"Du bist ein medizinischer Assistent für endoskopische Dokumentation.\n"
            f"Ordne folgende Lageangabe einer der vorgegebenen Kategorien zu.\n\n"
            f"Lageangabe: {location_out}\n\n"
            f"Regeln:\n"
            f"- IC-Klappe, Coecalpol, Appendix -> Zoekum\n"
            f"- Rechte oder linke Flexur, Colon transversum -> entsprechende Flexur oder Colon transversum\n"
            f"- Linea dentata, Analkanal, Ampulle, anorektal -> Rektum\n"
            f"- Rektosigmoidaler Uebergang -> Sigma\n"
            f"- Praefixe wie distal, proximal ignorieren\n"
            f"- Falls keine eindeutige Zuordnung moeglich -> Originalangabe unveraendert zurueckgeben\n"
            + json_suffix
        )

        ausgabe_text = None
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreiches medizinisches Assistenzsystem."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                timeout=1800,
            )
            ausgabe_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error in row {idx}: {e}")
            failed_rows.append((idx, str(e)))

        print("Output:\n", ausgabe_text)
        df.at[idx, "location_mapped_raw"] = ausgabe_text if ausgabe_text is not None else ""

        final_mapped = loc_str  # default fallback
        if ausgabe_text:
            parsed = extract_first_json_with_location(ausgabe_text)
            print("Parsed JSON:", parsed)
            if parsed is not None:
                raw_loc = parsed.get("location", "")
                if raw_loc.strip().lower() == "siehe_distanz" and re.search(r"\d", loc_str):
                    final_mapped = loc_str
                else:
                    normalized = normalize_location(raw_loc)
                    final_mapped = normalized if normalized is not None else raw_loc

        print("Response time:", time.time() - start, "seconds")

    # --- Save mapping ---
    df.at[idx, "location_mapped"] = final_mapped
    if df.at[idx, "location_mapped_raw"] == "":
        df.at[idx, "location_mapped_raw"] = loc_str

    # --- Check Lage/Distanz match against location_mapped ---
    match_flag, match_reason = location_distance_match(lage, distanz, final_mapped)
    df.at[idx, "location_distance_match"] = match_flag
    df.at[idx, "location_distance_match_reason"] = match_reason
    print(f"location_distance_match: {match_flag} | {match_reason}")

    processed_counter += 1
    save_if_due(processed_counter)

df.to_excel(OUTFILE, index=False)
end_tot = time.time()
print("Total run time:", end_tot - start_tot, "seconds")
print("\n====================")
print("DONE — FAILED ROWS")
print("====================")
for idx, err in failed_rows:
    print(f"Row {idx}: {err}")
