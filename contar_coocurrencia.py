import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("models/tflite_labels_v2.json", encoding="utf-8") as f:
    l2id = json.load(f)
id2l = {v: k for k, v in l2id.items()}

B_EDAD = l2id["B-EDAD"]
B_DUR  = l2id["B-DURACION"]

total = con_ambas = solo_edad = solo_dur = ninguna = 0

for split in ["train_v2", "val_v2", "test_v2"]:
    with open(f"models/tflite_{split}.json", encoding="utf-8") as f:
        data = json.load(f)
    for ej in data:
        labels = ej["label_ids"]
        tiene_edad = B_EDAD in labels
        tiene_dur  = B_DUR  in labels
        total += 1
        if tiene_edad and tiene_dur: con_ambas += 1
        elif tiene_edad:             solo_edad += 1
        elif tiene_dur:              solo_dur  += 1
        else:                        ninguna   += 1

print(f"Total ejemplos analizados : {total:>8,}")
print(f"Con B-EDAD Y B-DURACION   : {con_ambas:>8,}  ({con_ambas/total*100:.1f}%)")
print(f"Solo B-EDAD               : {solo_edad:>8,}  ({solo_edad/total*100:.1f}%)")
print(f"Solo B-DURACION           : {solo_dur:>8,}  ({solo_dur/total*100:.1f}%)")
print(f"Ninguna de las dos        : {ninguna:>8,}  ({ninguna/total*100:.1f}%)")
