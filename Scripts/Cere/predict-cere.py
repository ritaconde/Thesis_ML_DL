import pandas as pd
import predict
from Bio import SeqIO
import re
from sklearn.impute import SimpleImputer

def get_seq_description(seq_description):
    m = re.search(r"^(.*)\[", seq_description)
    description = str(m.group())[:-1].strip()

    return description

def get_sequence_description(seq_id, fasta_sequences):
    seq = ''
    for sequence in fasta_sequences:
        if sequence.id == seq_id:
            seq = get_seq_description(sequence.description)
            break

    return seq


def predict_sequences(sequences):
    features_data = pd.read_csv('Cere/InFeatures_FeaturesToPredict-cere.faa.csv', sep=',')

    print("Load models")
    models = predict.load_models('gb,knn,log,nb,rf,svm,tree')
    print("Finish load models")

    file_to_print = open(f"Cere/TransporterPrediction-cere.faa.csv", "w")

    predict.print_header(file_to_print, models.keys())

    i = 0
    for f in features_data.values:
        description = get_seq_description(sequences[i].description)

        features_values_transformed = SimpleImputer().fit_transform(f.reshape(1, -1))
        predictions = []
        for m in models.values():
            p = m.predict(features_values_transformed).tolist()
            predictions.append(','.join(str(v) for v in p))
        
        predict.print_row(file_to_print, description, predictions)

        i+=1

    file_to_print.close()


if __name__ == '__main__':
    fasta_sequences = list(SeqIO.parse('Cere/cere.faa', 'fasta'))

    predict_sequences(fasta_sequences)