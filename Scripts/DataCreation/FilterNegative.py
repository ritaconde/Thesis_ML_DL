import Utils.File as File
from Bio import SeqIO
import random
import DataCreation.Constants as Constants

negative_main_path = Constants.NEGATIVE_PATH

def get_negative_records():
    paths = negative_main_path.copy()
    paths.append(Constants.NEGATIVE_DATA_FILE)
    negative_file_path = File.get_file_path(paths)

    # Get list of SeqRecords
    handle = open(negative_file_path)
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    return records

def generate_output_file(records):

    paths = negative_main_path.copy()
    paths.append(Constants.NEGATIVE_FILTERED_DATA_FILE)
    negative_file_path_filtered = File.get_file_path(paths)
    File.check_path(negative_file_path_filtered)

    file_pointer=open(negative_file_path_filtered, "w")

    for r in records:
        file_pointer.write(str(r.format("fasta")))

    file_pointer.close()

def generate_percentage_dataset(positive_count):
    '''
    Generation of negative set that represents 90% of the positive cases dataset
    Also, a subdataset that represents 35% of previsouly generated dataset
    '''

    records = get_negative_records()
    negative_count = len(records)

    negative_representation = int((positive_count * 90) / 10)
    final_negative = int((negative_representation * 35) / 100)

    filtered_records = records

    if final_negative < negative_count:
        filtered_records = random.sample(records, final_negative)

    # Output to file
    generate_output_file(filtered_records)

    number_records = len(filtered_records)

    print(f"Percentage Negative file created with {number_records} records")

    return number_records


def generate_random_dataset(positive_count):
    '''
    Randomly generation of negative set
    '''

    records = get_negative_records()

    # Random select of # records
    filtered_records = random.sample(records, positive_count)

    # Output to file
    generate_output_file(filtered_records)

    number_records = len(filtered_records)

    print(f"Random Negative file created with {number_records} records")

    return number_records

