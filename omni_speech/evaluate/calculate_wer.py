from jiwer import wer, cer
import json
import re, string
from transformers import pipeline
from tqdm import tqdm
from spellchecker import SpellChecker
import urllib.request


# 1. Euskarazko hitz zerrenda deskargatu (adibide gisa, Githubetik)
# Zerrenda hau oinarrizkoa da, baina hitz asko ditu
url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/eu/eu_50k.txt"
response = urllib.request.urlopen(url)
hitzak_raw = response.read().decode('utf-8').splitlines()
euskal_hitzak = [line.split()[0] for line in hitzak_raw]
spell = SpellChecker(language=None)  # Ez dugu ingelesa nahi
spell.word_frequency.load_words(euskal_hitzak)

# Maiztasun fitxategia denez, hitz bakoitzaren ondoan zenbaki bat dator; garbitu egingo dugu
euskal_hitzak = [line.split()[0] for line in hitzak_raw]

JSON_PATH = "/home/asudupe/Latxa-Omni/results/euskaraz_bakarrik_2.json"
MODEL_PATH = "HiTZ/cap-punct-eu"
translator = pipeline(task="translation", model=MODEL_PATH, tokenizer=MODEL_PATH, device='cuda')    


def normalize(text):
    text = re.sub(r'\.(?!\s)', '. ', text)
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def puntuazioa(text):
    return translator(text)[0]['translation_text']


# Euskarazko hiztegia kargatu (lehenago sisteman instalatuta izan behar duzu)

def esaldia_iragazi(esaldia, onartutako_erratu_max = 0):
    # Testua garbitu (puntuazioa kendu)
    hitzak = re.findall(r'\b\w+\b', esaldia.lower())
    
    if not hitzak:
        return False

    # Begiratu zein hitz ez dauden euskal hiztegian
    nec_hitzak = spell.unknown(hitzak)
    
    # Izen bereziak (Normalean letra larriz hasten direnak) askotan "erratu" gisa agertzen dira
    # Horiek baztertzeko logika gehiago gehitu daiteke hemen
    
    # print(f"Ezezagunak: {nec_hitzak}")
    
    # Esaldia mantendu hitz arrotz kopurua muga baino txikiagoa bada
    return len(nec_hitzak) <= onartutako_erratu_max

def hitz_arrotza_da(esaldia):
    # 1. Hizki arrotzak (Euskal alfabetoan ez daudenak: q, v, y eta batzuetan w)
    # Hemen 'w' gehitu dugu, askotan maileguetan agertzen delako
    hizki_arrotzak = r'[qvywñ1234567890]'
    
    # 2. Karaktere konbinazio arrotzak (euskaran ohikoak ez direnak)
    # ck (back, track), sh (show), ch (check), ph (phone), th (thing)
    konbinazio_arrotzak = r'ck|sh|ch|ph|th|ll|aa|ee|ii|oo|uu' 
    # Oharra: 'll' kendu dezakezu izen berezi euskaldunak badituzu (adib. 'Llanos')

    # Esaldia xehez jarri azterketa errazteko
    esaldia_lower = esaldia.lower()

    # Bilatu hizki edo konbinazio horiek
    if re.search(hizki_arrotzak, esaldia_lower) or re.search(konbinazio_arrotzak, esaldia_lower):
        return True # Arrotza da
    
    return False # Euskara garbia dirudi

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

out_text = ""
transcribed_text = ""
texts = {}
texts['global'] = {"text": "", "transcribed": "", "kopurua": 0}
# euskaraz_bakarrik = []
for d in tqdm(data):
    # if esaldia_iragazi(d['response_text_llm']) and not hitz_arrotza_da(d['response_text_llm']):
    # if not hitz_arrotza_da(d['response_text_llm']):
        # euskaraz_bakarrik.append(d)
        try:
            a = d['response_asr']
        except:
            break
        if d['split'] not in texts:
            texts[d['split']] = {"text": "", "transcribed": "", "kopurua": 0}
        texts[d['split']]["text"] += d['response_text_llm'] + " "
        texts[d['split']]["transcribed"] += puntuazioa(d['response_asr']) + " "
        texts[d['split']]["kopurua"] += 1
        texts['global']["text"] += d['response_text_llm'] + " "
        texts['global']["transcribed"] += puntuazioa(d['response_asr']) + " "
        texts['global']["kopurua"] += 1

wer_by_split = {}
cer_by_split = {}
for split in texts.keys():
    wer_by_split[split] = wer(texts[split]["text"], texts[split]["transcribed"])
    cer_by_split[split] = cer(texts[split]["text"], texts[split]["transcribed"])

print("\n" + "="*70)
print("WER ETA CER EMAITZAK SPLIT BAKOITZEKO NEMO PUNTUAZIOA")
print("="*70)

for split, w_errors in wer_by_split.items():
    print(f"Split: {split:<20} | WER: {w_errors:.4f} | CER: {cer_by_split[split]:.4f} | Kopurua: {texts[split]['kopurua']}")

# with open("/home/asudupe/Latxa-Omni/results/euskaraz_bakarrik_3.json", 'w') as f:
#     json.dump(euskaraz_bakarrik, f, indent=4)

# total_wer_errors = []
# total_cer_errors = []

# for split, w_errors in wer_by_split.items():
#     # avg_wer = sum(w_errors) / len(w_errors)
    
#     c_errors = cer_by_split.get(split, [])
#     if c_errors:
#         avg_cer = sum(c_errors) / len(c_errors)
#     else:
#         avg_cer = 0.0

#     print(f"Split: {split:<20} | Samples: {len(w_errors):<5} | WER: {w_errors:.4f} | CER: {avg_cer:.4f}")
    
#     total_wer_errors.extend(w_errors)
#     total_cer_errors.extend(c_errors)

# if total_wer_errors:
#     global_wer = sum(total_wer_errors) / len(total_wer_errors)
    
#     global_cer = 0.0
#     if total_cer_errors:
#         global_cer = sum(total_cer_errors) / len(total_cer_errors)

#     print("-" * 70)
#     print(f"Global WER: {global_wer:.4f}")
#     print(f"Global CER: {global_cer:.4f}")
# print("="*70)

