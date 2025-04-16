import stanza

# Download Hindi model only once
# stanza.download('hi')  # Uncomment only on first run

# Initialize the pipeline
nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos')

# Load sentences from file
with open("hindi_testing.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# POS Tagging for each sentence
for idx, line in enumerate(lines, start=1):
    line = line.strip()
    if not line:
        continue  # Skip empty lines

    doc = nlp(line)

    print(f"\n📌 Sentence {idx}: {line}")
    print("शब्द\t\tUniversal POS\tDetailed POS")
    print("-----\t\t-------------\t------------")
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f"{word.text}\t\t{word.upos}\t\t{word.xpos}")
