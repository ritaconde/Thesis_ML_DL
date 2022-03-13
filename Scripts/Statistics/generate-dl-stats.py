import pandas as pd
from Bio import SeqIO
import Utils.calc as Calculate

FILE = "Ecoli/ecoli-hybrid.csv"
ORIGINAL_FILE = 'Ecoli/Ecoli.faa'
NAME = "Ecoli"

# FILE = "Cere/cere-hybrid.csv"
# ORIGINAL_FILE = 'Cere/cere.faa'
# NAME = "Cerevisiae"

THRESHOLD = 0.7

def get_sequence_by_id(seq_id, sequence_list):
    for s in sequence_list:
        if s.id == seq_id:
            return s

def is_transporter(sequence, protein_tcdb_sequences):
    for psequence in protein_tcdb_sequences:
        if sequence.seq == psequence.seq:
            return True

    return False

data = pd.read_csv(FILE, sep=';', header=None).to_numpy()
protein_tcdb_sequences = list(SeqIO.parse('/Users/marcelo/Desktop/Rita/Tese/tese.nosync/Data/Positive/positive-filtered.fasta', 'fasta'))
fasta_sequences = list(SeqIO.parse(ORIGINAL_FILE, 'fasta'))

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for row in data:
    is_predict_transporter = False
    sequence_id = row[0]
    if row[1] >= THRESHOLD:
        is_predict_transporter = True

    sequence = get_sequence_by_id(sequence_id, fasta_sequences)

    is_sequence_transporter = is_transporter(sequence, protein_tcdb_sequences)

    if  is_predict_transporter and is_sequence_transporter:
        true_positive +=1

    elif is_predict_transporter and not is_sequence_transporter:
        false_positive +=1

    elif not is_predict_transporter and is_sequence_transporter:
        false_negative +=1

    else:
        true_negative +=1

Calculate.calculate_data(NAME, true_positive, true_negative, false_positive, false_negative)

