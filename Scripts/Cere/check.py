from Bio import SeqIO
import pandas as pd
import re
import math

def get_seq_description(seq_description):
    m = re.search(r"^(.*)\.\d", seq_description)
    description = str(m.group()).strip()

    return description

fasta_sequences = list(SeqIO.parse('cere.faa', 'fasta'))
protein_tcdb_sequences = list(SeqIO.parse('/Users/marcelo/Desktop/Rita/Tese/tese.nosync/Data/Positive/positive-filtered.fasta', 'fasta'))

transporters_cere = []

for p in protein_tcdb_sequences:
    for f in fasta_sequences:    
        if str(f.seq) == str(p.seq):
            transporters_cere.append(f.id)
            break


transporters_cere = list(set(transporters_cere))

features_data = pd.read_csv('TransporterPrediction-cere.faa.csv', sep=',')
count = 0
transp = []
columns = features_data.columns[1:]

vp = {elem: 0 for elem in columns}
vn = {elem: 0 for elem in columns}
fp = {elem: 0 for elem in columns}
fn = {elem: 0 for elem in columns}

for fdata in features_data.values:
    fdata_id = get_seq_description(fdata[0])

    if fdata_id in transporters_cere: # Aqui calcula com base na condição de ser positivo - transportador
        for p in range(len(columns)):
            column_name = columns[p]
            column_index = p+1

            if str(fdata[column_index]) == '1': # verifica se predição foi que é transportador - incrementa verdadeiro positivo
                vp[column_name] +=1
            else:
                fn[column_name] +=1 # predição deu não transportador - incrementa falso negativo

    else:  # Aqui calcula com base na condição de ser negativo - não é transportador
        for p in range(len(columns)):
            column_name = columns[p]
            column_index = p+1
            
            if str(fdata[column_index]) == '1': # verifica se predição foi que é transportador - incrementa falso positivo
                fp[column_name] +=1
            else:
                vn[column_name] +=1 # predição deu não transportador - incrementa verdadeiro negativo

print("Cerevisae matriz de confusão valores por modelo")
print(f"Total de sequencias: {len(fasta_sequences)}")
print(f"Total de sequencias transportadoras: {len(transporters_cere)}")

print("Verdadeiros Positivos")
print(vp)
print("Verdadeiros Negativos")
print(vn)
print("Falsos Positivos")
print(fp)
print("Falsos Negativos")
print(fn)

for model in columns:
    model_vp = vp[model]
    model_fp = fp[model]
    model_vn = vn[model]
    model_fn = fn[model]


    up = (model_vp * model_vn) - (model_fp * model_fn)
    down = (model_vp + model_fn) * (model_vp + model_fp) * (model_vn + model_fp) * (model_vn + model_fn)

    ccm = up / math.sqrt(down)
    acc = (model_vp + model_vn) / (model_vp + model_vn + model_fp + model_fn)
    prec = model_vp / (model_vp+model_fp)
    recall = model_vp / (model_vp + model_fn)
    f1 = 2* ((prec*recall) / (prec + recall))

    print(f"{model}")
    print(f"CCM: {ccm}")
    print(f"ACC: {acc}")
    print(f"PRE: {prec}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")