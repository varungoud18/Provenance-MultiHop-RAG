import re

# Edge cases:
# 1. Citation before period: "The model achieved 0.364 [1]."
# 2. Citation after period: "The model achieved 0.364. [1]"
# 3. Multiple citations: "Used dataset [1][2]."
# 4. Decimals and model names: "Tested GPT-3.5 with lr 0.001 [2]."

EDGE_CASE_TEXT = """* **Performance:** The model tested was GPT-3.5-turbo [1]. It scored 0.364 on ROUGE-1. [2]
* **Conclusion:** This shows promise for RAG systems [1][3]."""

def robust_sentence_split(line: str):
    # Step 1: If citation appears immediately after period (e.g. '. [1]' or '.[1]'), 
    # normalize it so citation sits before punctuation or is bound to the preceding sentence
    normalized = re.sub(r'\.\s*(\[\d+\]+)', r' \1.', line)
    
    # Step 2: Split sentences: period/exclamation/question followed by space and capital letter or bullet
    # Exclude periods in numbers: negative lookbehind for digit, negative lookahead for digit
    pattern = r'(?<!\d)(?<=[.!?])\s+(?=[A-Z*#])'
    sentences = re.split(pattern, normalized)
    return sentences

print("Normalized edge-case test:")
for line in EDGE_CASE_TEXT.strip().split("\n"):
    for s in robust_sentence_split(line):
        cits = re.findall(r'\[(\d+)\]', s)
        print(f"  Sentence: {s.strip()} | Citations: {cits}")
