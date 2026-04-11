import os
import time
from openai import OpenAI
import pandas as pd

with open("prompt.txt", "r", encoding="utf-8") as f:
    user_prompt = f.read().strip()

client = OpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key=os.environ.get("OLLAMA_API_KEY", "dummy-key"),
)
MODEL = os.environ["OLLAMA_MODEL"]

models = client.models.list()
for m in models.data:
    print(m.id)

df = pd.read_excel("input_data.xlsx")
OUTFILE = "llm_out_stage1.xlsx"
# Ensure the output column exists
if "llm_out" not in df.columns:	
    df["llm_out"] = None

failed_rows = []
json_suffix = ""


start_tot = time.time()
processed_counter = 0
SAVE_EVERY = 10


for i, (idx, row) in enumerate(df.iterrows(), start=1):
    # Skip rows that already have output (useful for resuming)
    if pd.notna(df.at[idx, "llm_out"]):
        print(f"Skipping row {idx} (already has llm_out).")
        continue

    start = time.time()
    report = row["report"]
    print(f"{idx}. Report:", report)

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreiches Assistenzsystem."},
                {"role": "user", "content": user_prompt + "\n\nHier kommt der neue report:\n" + report + json_suffix},
            ],
            temperature=0.2,
            timeout=1800,
        )
        llm_text = response.choices[0].message.content
        print(f"Row {idx} result:")
        print(llm_text)
        df.at[idx, "llm_out"] = llm_text

    except Exception as e:
        print(f"⚠️ Error in row {idx}: {e}")
        print("→ Skipping this row.")
        failed_rows.append((idx, str(e)))
        df.at[idx, "llm_out"] = "Error"

    processed_counter += 1

    # Persist every SAVE_EVERY processed rows
    if processed_counter % SAVE_EVERY == 0:
        try:
            print("Trying to save file")
            df.to_excel(OUTFILE, index=False)
            print(f"Saved progress to {OUTFILE} after {processed_counter} processed rows.")
        except Exception as save_err:
            print(f"⚠️ Error saving file after {processed_counter} rows: {save_err}")

    end = time.time()
    print("Response time:", end - start, "seconds\n")

# Final save
try:
    df.to_excel(OUTFILE, index=False)
    print(f"Final save to {OUTFILE}.")
except Exception as save_err:
    print(f"⚠️ Error during final save: {save_err}")

end_tot = time.time()
print("Total run time:", end_tot - start_tot, "seconds")

print("\n====================")
print("DONE — FAILED ROWS")
print("====================")
for idx, err in failed_rows:
    print(f"Row {idx}: {err}")
