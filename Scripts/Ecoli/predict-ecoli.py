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

def get_joined_sequences(fasta_sequences):
    base_data = pd.read_csv('Ecoli/joined.csv', sep=',')

    for data in base_data.values:
        description = get_sequence_description(data[0], fasta_sequences)
        base_data['Sequence ID'] = base_data['Sequence ID'].replace(data[0], description)
    
    return base_data

def predict_sequences(sequences):
    features_data = pd.read_csv('Ecoli/InFeatures_FeaturesToPredict-Ecoli.faa.csv', sep=',')
    result = []

    print("Load models")
    models = predict.load_models('gb,knn,log,nb,rf,svm,tree')
    print("Finish load models")

    i = 0
    for f in features_data.values:
        aux = [get_seq_description(sequences[i].description)]

        features_values_transformed = SimpleImputer().fit_transform(f.reshape(1, -1))

        for m in models.values():
            prediction = m.predict(features_values_transformed).tolist()
            
            aux.append(','.join(str(x) for x in prediction))

        result.append(aux)
        i+=1

    return result


if __name__ == '__main__':
    fasta_sequences = list(SeqIO.parse('Ecoli/Ecoli.faa', 'fasta'))
    
    print("Get joined sequences")
    joined = get_joined_sequences(fasta_sequences)

    print("Predict missing sequences")
    rest = predict_sequences(fasta_sequences[joined.shape[0]:])

    for r in rest:
        a_series = pd.Series(r, index = joined.columns)
        joined = joined.append(a_series, ignore_index=True)

    joined.to_csv('Ecoli/ecoli-final.csv', sep=',', index=False)
