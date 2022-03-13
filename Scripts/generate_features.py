import sys
import joblib
from Bio import SeqIO
import numpy as np
import InDatasetCreation.InFeaturesCreation as in_features_creation
import DataCreation.CreateCsv as data_csv
import Utils.File as File


def main(argv):

    if len(argv) < 1 or not File.files_exists([argv[0]]):
        print("Please, insert a valid fasta file path")
        return
    
    file_path = argv[0]
    f = open(file_path)

    filename = File.get_file_name(f.name)

    file_to_print = f"FeaturesToPredict-{filename}.csv"

    fasta_sequences = list(SeqIO.parse(f, 'fasta'))

    total_sequences = len(fasta_sequences)
    i = 1
    features = np.array(in_features_creation.GetFeaturesColumns())
    for fasta_sequence in fasta_sequences:

        succ = False

        while not succ:
            try:
                print(f"Sequence: {i} of {total_sequences}")
                sequence_pre_data = data_csv.get_default_record_data(fasta_sequence)
                features_values = in_features_creation.GetFeaturesValues(sequence_pre_data)                
                features = np.vstack((features, features_values))
                in_features_creation.print_file(features, file_to_print)
                succ = True
                i+=1
            except Exception as e:
                print("Error detected: {0}".format(e))
                succ = False

if __name__ == '__main__':
    main(sys.argv[1:])