import Utils.File as File
import InDatasetCreation.Constants as Constants
import InDatasetCreation.Features.Aminoacid as Aminoacid
import InDatasetCreation.Features.AlphaPhobius as AlphaPhobius
import InDatasetCreation.Features.BetaBomp as BetaBomp
import InDatasetCreation.Features.SubCellularLocTree as SubCellularLocTree
import InDatasetCreation.Features.RelatedDomains as RelatedDomains
import pandas as pd
import numpy as np

def print_file(features, original_filename):

    paths = Constants.MAIN_DATA_PATH.copy()

    paths.append(Constants.INFEATURES_FOLDER)
    paths.append(f"InFeatures_{original_filename}")
    data_file_path = File.get_file_path(paths)
    File.check_path(data_file_path)

    dataframe = pd.DataFrame(features[1:], index=None, columns=features[0])
    dataframe.to_csv(data_file_path, sep=',', index=False)

def CreateInFeatures(filename):
    paths = Constants.MAIN_DATA_PATH.copy()
    paths.append(filename)
    data_file_path = File.get_file_path(paths)

    data = pd.read_csv(data_file_path,sep=",")
    total = len(data)
    index = range(total)

    features = np.array(GetFeaturesColumns())

    for i in index:
        print(f"Process sequence {i+1} of {total}")

        data_row = data.values[i]
        succ = False

        while not succ:
            try:
                data_row_features = GetFeaturesValues(data_row)            
                features = np.vstack((features, data_row_features))
                print_file(features, filename)
                succ = True
            
            except Exception as e:
                print("Error detected: {0}".format(e))
                succ = False

def GetFeaturesColumns():
    ''' Get columns names for features dataset '''
    columns_names = []

    columns_names += Aminoacid.get_aminoacid_composition_columns()
    columns_names += Aminoacid.get_aminoacid_occurence_columns()
    columns_names += Aminoacid.get_aminoacid_physico_chemical_composition_columns()
    columns_names += Aminoacid.get_aminoacid_physico_chemical_occurrence_columns()
    columns_names += Aminoacid.get_dipeptide_composition_columns()
    columns_names += RelatedDomains.get_related_domains_columns()
    columns_names += AlphaPhobius.get_alpha_columns()
    columns_names += BetaBomp.get_beta_columns()
    columns_names += SubCellularLocTree.get_subcellular_columns()

    return columns_names

def GetFeaturesValues(data_row):
    ''' Get features values for each sequence data row '''

    features_values = Aminoacid.create_aminoacid_composition(data_row)
    features_values = np.hstack((features_values, Aminoacid.create_aminoacid_occurrence(data_row)))
    features_values = np.hstack((features_values, Aminoacid.create_aminoacid_physico_chemical_composition(data_row)))
    features_values = np.hstack((features_values, Aminoacid.create_aminoacid_physico_chemical_occurrence(data_row)))
    features_values = np.hstack((features_values, Aminoacid.create_dipeptide_composition(data_row)))
    features_values = np.hstack((features_values, RelatedDomains.create_related_domains(data_row)))
    features_values = np.hstack((features_values, AlphaPhobius.create_num_alphahelices_and_signalpeptide(data_row)))
    features_values = np.hstack((features_values, BetaBomp.create_betabarrels(data_row)))
    features_values = np.hstack((features_values, SubCellularLocTree.create_subcellular_location(data_row)))

    return features_values
