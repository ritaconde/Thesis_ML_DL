import DataCreation.FilterNegative as filter_negative
import DataCreation.FilterPositive as filter_positive
import DataCreation.CreateCsv as data_csv
import InDatasetCreation.InFeaturesCreation as in_features_creation

def generate_filtered_data():
    '''
    Main function to generate positive and negative data
    '''


    # Split positive and negative data
    positive_cases = filter_positive.generate_positive_data()
    #negative_cases = filter_negative.generate_random_dataset(positive_cases)
    # negative_cases = filter_negative.generate_percentage_dataset(positive_cases)

    # Generate final csv for training data - mixed between positive and negative cases
    # data_csv.generate_data_csv()

def generate_input_features():
    in_features_creation.CreateInFeatures("data-filtered-random.csv")

    

if __name__ == '__main__':
    generate_filtered_data()
    # generate_input_features()

    

