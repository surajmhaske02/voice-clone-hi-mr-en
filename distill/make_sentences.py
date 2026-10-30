"""Build clean, speakable sentence lists (en / hi / mr) from Wikipedia + everyday phrases.

Writes distill/sentences/{en,hi,mr}.txt. Only one parquet row group per language is streamed.
Usage: .venv-omni/bin/python distill/make_sentences.py [--n 1200]
"""
import argparse, os, random, re
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = "datasets/wikimedia/wikipedia/20231101.{}/train-00000-of-{}.parquet"
SHARDS = {"en": "00041", "hi": "00002", "mr": "00001"}

# Everyday lines so the voice also sounds natural in conversation, not only "encyclopedia voice".
EVERYDAY = {
    "en": ["Hello, how are you today?", "Good morning, did you sleep well?", "Please call me when you reach home.",
           "I will be there in ten minutes.", "Thank you so much for your help.", "What would you like to eat tonight?",
           "Don't forget to take your medicine.", "The meeting has been moved to tomorrow.",
           "Can you please turn off the lights?", "Happy birthday, have a wonderful year!",
           "It is raining heavily outside, take an umbrella.", "Sorry, I did not hear you. Can you repeat that?"],
    "hi": ["नमस्ते, आप कैसे हैं?", "सुप्रभात, क्या आप अच्छे से सोए?", "घर पहुँचकर मुझे फ़ोन करना।",
           "मैं दस मिनट में वहाँ पहुँच जाऊँगा।", "आपकी मदद के लिए बहुत धन्यवाद।", "आज रात आप क्या खाना चाहेंगे?",
           "दवाई लेना मत भूलना।", "मीटिंग कल तक के लिए टल गई है।", "कृपया लाइट बंद कर दीजिए।",
           "जन्मदिन मुबारक हो, आपका साल शानदार रहे!", "बाहर तेज़ बारिश हो रही है, छाता ले जाना।",
           "माफ़ कीजिए, मैंने सुना नहीं। क्या आप दोबारा बोल सकते हैं?"],
    "mr": ["नमस्कार, तुम्ही कसे आहात?", "सुप्रभात, झोप चांगली झाली का?", "घरी पोहोचल्यावर मला फोन कर.",
           "मी दहा मिनिटांत तिथे पोहोचतो.", "तुमच्या मदतीबद्दल खूप धन्यवाद.", "आज रात्री जेवायला काय हवं?",
           "औषध घ्यायला विसरू नकोस.", "मीटिंग उद्यापर्यंत पुढे ढकलली आहे.", "कृपया दिवे बंद करा.",
           "वाढदिवसाच्या हार्दिक शुभेच्छा, तुमचं वर्ष छान जावो!", "बाहेर जोरात पाऊस पडतोय, छत्री घेऊन जा.",
           "माफ करा, मला ऐकू आलं नाही. पुन्हा सांगाल का?"],
}

SPLIT = re.compile(r"(?<=[.!?।])\s+")
OK = {
    "en": re.compile(r"^[A-Za-z ,.'!?-]+$"),
    "hi": re.compile("^[\\u0900-\\u0963\\u0970-\\u097F ,.!?\\u0964-]+$"),  # Devanagari letters only (no digits, no Latin)
    "mr": re.compile("^[\\u0900-\\u0963\\u0970-\\u097F ,.!?\\u0964-]+$"),
}


def good(s, lang):
    s = s.strip()
    w = s.split()
    return (5 <= len(w) <= 18 and 25 <= len(s) <= 140 and OK[lang].match(s) and s[-1] in ".!?।"
            and not re.search(r"\b[A-Z]{2,}\b", s) and "  " not in s and s.count(",") <= 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="sentences per language")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "sentences"), exist_ok=True)
    fs = HfFileSystem()
    random.seed(0)
    for lang, n in SHARDS.items():
        with fs.open(SHARD.format(lang, n), "rb") as f:
            pf = pq.ParquetFile(f)
            texts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
        seen, out = set(), []
        for t in texts:
            for para in t.split("\n"):
                for s in SPLIT.split(para):
                    s = re.sub(r"\s*\([^)]*\)", "", s).strip()  # drop parentheticals
                    if good(s, lang) and s not in seen:
                        seen.add(s)
                        out.append(s)
        random.shuffle(out)
        out = EVERYDAY[lang] + out[: a.n - len(EVERYDAY[lang])]
        with open(os.path.join(HERE, "sentences", f"{lang}.txt"), "w") as f:
            f.write("\n".join(out) + "\n")
        print(lang, len(out), "sentences, e.g.:", out[len(EVERYDAY[lang])], "|", out[-1])


if __name__ == "__main__":
    main()
