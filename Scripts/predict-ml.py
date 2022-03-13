import sys
import joblib
from Bio import SeqIO
import InDatasetCreation.InFeaturesCreation as in_features_creation
import DataCreation.CreateCsv as data_csv
import Utils.File as File
from sklearn.impute import SimpleImputer

models_names = {
        'gb': "GB_model.pkl",
        'knn': "Knn_model.pkl",
        'log': "Logistic_model.pkl",
        'nb': "NB_model.pkl",
        'rf': "RF_model.pkl",
        'tree': "Tree_model.pkl",
        'svm': "SVM_model.pkl"
    }

models_path = "/Volumes/TOSHIBA/tese/"

def load_models(models):
    result = {}

    models_list = models.split(',')

    for m in models_list:

        if m not in models_names:
            print(f"Can't load model: {m}")
            exit()

        model = joblib.load(f"{models_path}{models_names[m]}")
        result[m] = model

    print("All models are loaded")

    return result

def print_header(file, models_names):

    header_list = ["Sequence ID"]

    header_list.extend(models_names)

    header = [','.join(header_list) + '\n']
    file.writelines(header)


def print_row(file, sequence_id, predictions):

    row_list = [sequence_id]
    row_list.extend(predictions)

    row = [','.join(row_list) + '\n']

    file.writelines(row)


def main(argv):

    if len(argv) < 2 or not File.files_exists([argv[0]]):
        print("Please, insert a valid fasta file path or a valid model type list (gb, knn, log, nb, rf, svm, tree)")
        return
    
    models = load_models(argv[1])

    file_path = argv[0]
    f = open(file_path)

    filename = File.get_file_name(f.name)

    file_to_print = open(f"TransporterPrediction-{filename}.csv", "w")

    print_header(file_to_print, models.keys())

    fasta_sequences = list(SeqIO.parse(f, 'fasta'))
    
    for fasta_sequence in fasta_sequences:
        sequence_pre_data = data_csv.get_default_record_data(fasta_sequence)

        features_values = in_features_creation.GetFeaturesValues(sequence_pre_data)
        features_values_transformed = SimpleImputer().fit_transform(features_values.reshape(1, -1))

        predictions = []
        for model in models.values():
            prediction = model.predict(features_values_transformed).tolist()
            final_prediction = []
            for p in prediction:
                final_prediction.append(str(p))

            predictions.append(','.join(final_prediction))

        print_row(file_to_print, sequence_pre_data[0], predictions)

    file_to_print.close()

if __name__ == '__main__':
    main(sys.argv[1:])