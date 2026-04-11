import os
import pandas as pd
import json
import re
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key=os.environ.get("OLLAMA_API_KEY", "dummy-key"),
)
MODEL = os.environ["OLLAMA_MODEL"]

FEWSHOT = """
Beispiel 1:
Report:
unauffällig. Im Sigma Polypenknospe, PE II. Glatte Einführung bis in das Zökum. hier findet sich hinter der Klappe ein 60 x 90 mm durchmessender adenoider Tumor. Randomisatiojn auf STEP in GG 3. Gutes Anheben auf Unterspritzung mit HydroJet. Fraktionierte komplette Resektion, keine Blutung. Kleine Reste werden mit APC versorgt, bis makroskopisch kein Tumor mehr verblieben ist. Am Rand der Resektionsfläche entsteht ein schlitzförmiger Muscularisdefekt, der mit 4 Clips verschlossen wird, Abpunktieren freier Luft. Bergung im Netz (I)

Erwartete Ausgabe:
- Zökum; 60 × 90 mm; fraktioniert

Beispiel 2:
Report:
Vorspiegeln bis zum unauffälligem Ende des Hartmannstumpfes bei 28 cm a.a. bei 18 cm a.a. das ca. 1 x 1 cm große Adenom, welches nach Unterspritzen mit HAES/TB in toto entfernt wird. durch das Stoma Vorspiegeln bis zum Coecalpol, dort zeigt sich ein kleines Adenom im Coecalpol, welches mit der Zange entfernt wird (PE I). Ein weiteres ca. 1,5 x 1 cm großes flaches Adenom gegenüber der IC-Klappe wird biopsiert und belassen (PE II). Direkt auf der IC-Klappe zeigt sich ein ca. 3 x 2 cm großes suspektes Adenom, welches lediglich biopsiert wird (PE III). Im C. ascendens finden sich in kleinem Abstand 3 aufsitzende Adenome von ca. 5 mm, 8 mm und 10 mm Größe, welche nach Aufspritzen mit HAES/TB mit der Schlinge in toto entfernt werden. (PE IV). Die Adenome werden mit dem Netz geborgen. Ein weiteres Adenom im C. transversum gut 1 cm messend wird im Rückzug belassen. Das restliche Kolon stellt sich als unauffällig dar.

Erwartete Ausgabe:
- 18 cm a.a.; 1 × 1 cm; en-bloc
- Coecalpol; nicht angegeben; en-bloc
- Colon ascendens; 5 mm; en-bloc
- Colon ascendens; 8 mm; en-bloc
- Colon ascendens; 10 mm; en-bloc
"""

JSON_SUFFIX = """
Gib zwei Felder zurück:
1. "reasoning": eine kurze Zusammenfassung deiner Überlegungen (kein <think>)
2. "answer": die finale Antwort als String, Einträge mit \\n getrennt
Beide Felder in gültigem JSON.
"""

def extract_answer_from_parsed(data: dict):
    """Return answer if present and non-empty, else None."""
    answer = data.get("answer")
    if answer and isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None

def try_parse_answer(text: str):
    """Try all parsing strategies. Returns answer string or None."""
    if not isinstance(text, str):
        return None

    # Remove <think> blocks and code fences
    text = re.sub(r"[\[\]]", "", text)
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text_clean = re.sub(r"```(?:json)?\s*", "", text_clean)
    text_clean = re.sub(r"```", "", text_clean)

    # 1. Strict JSON parse
    try:
        data = json.loads(text_clean)
        answer = extract_answer_from_parsed(data)
        if answer:
            return answer
    except json.JSONDecodeError:
        pass

    # 2. Find first {...} block containing "answer"
    candidates = re.findall(r"\{.*?\}", text_clean, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            answer = extract_answer_from_parsed(obj)
            if answer:
                return answer
        except:
            continue

    # 3. Regex direct capture of "answer" value
    match = re.search(r'"answer"\s*:\s*"(.*?)"(?=\s*[,}])', text_clean, flags=re.DOTALL)
    if match:
        return match.group(1).replace("\\n", "\n").strip()

    return None

def llm_repair(text: str):
    """Call LLM to reformat the malformed output into valid JSON."""
    print(f"Repair attempt for:\n{text}\n")

    prompt = f"""
Der folgende Text ist eine fehlerhafte oder unvollständige Ausgabe eines Sprachmodells.
Extrahiere daraus alle entfernten Polypen-/Tumorbefunde und gib sie im folgenden Format zurück:
- Lokalisation; Größe; Resektionsart

Nur vollständig oder fraktioniert entfernte Adenome berücksichtigen. Biopsierte oder belassene ignorieren.

Nutze folgende Beispiele als Orientierung:
{FEWSHOT}

Fehlerhafter Text:
{text}

{JSON_SUFFIX}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    repaired = response.choices[0].message.content.strip()
    print(f"Repair result:\n{repaired}\n")
    return try_parse_answer(repaired)

def clean_and_extract_answer(text: str):
    """Main entry: try parsing, fall back to LLM repair if needed."""
    answer = try_parse_answer(text)

    if not answer:
        print(f"Parsing failed, calling LLM repair...")
        answer = llm_repair(text)

    if not answer:
        print(f"Repair also failed.")

    return answer


df = pd.read_excel("llm_out_stage1.xlsx")
df["answer"] = df["llm_out"].map(clean_and_extract_answer)
df.to_excel("llm_out_stage2.xlsx", index=False)
print("Done. File written: llm_out_stage2.xlsx")
