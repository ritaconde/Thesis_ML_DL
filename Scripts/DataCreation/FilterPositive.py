import Utils.File as File
from Bio import SeqIO
import DataCreation.Constants as Constants

positive_main_path = Constants.POSITIVE_PATH

def reverse(text):
    """
    Returns the text backwards
    ex: input = text; output = txet
    """
    return (text[::-1])
        
def get_TC_ids(records):
    """
    Iterates through every record in the records file 
    and returns a list with each record TC_ID
    """
    TC_IDs=[]
    for record in records:
        id=str(record.id)
        rev=""
        i=len(id)-1
        while id[i] != "|":
            rev+= id[i]
            i-=1
        TC_IDs.append(reverse(rev))
    
    return (TC_IDs)

def filter_positive_records(records, tc_ids):
    """
    Filter positive records by TC classication category
    """

    ids_to_ignore = ['8', '9']
    i = 0

    filtered_records = []
    for r in records:
        if tc_ids[i][0] not in ids_to_ignore:
            filtered_records.append(r)
        i+=1

    return filtered_records

def generate_positive_data():
    """
    Generate the treated and filtered positives cases to be used on dataset creation
    """

    paths = positive_main_path.copy()
    paths.append(Constants.POSITIVE_DATA_FILE)
    positive_file_path = File.get_file_path(paths)

    # Get list of SeqRecords
    handle = open(positive_file_path)
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    # Get tc ids
    tc_ids = get_TC_ids(records)

    # Filter list of SeqRecords
    filtered_records = filter_positive_records(records, tc_ids)

    # Output to file
    paths = positive_main_path.copy()
    paths.append(Constants.POSITIVE_FILTERED_DATA_FILE)
    positive_file_path_filtered = File.get_file_path(paths)
    File.check_path(positive_file_path_filtered)

    file_pointer=open(positive_file_path_filtered, "w")

    for r in filtered_records:
        file_pointer.write(str(r.format("fasta")))

    file_pointer.close()

    number_positive = len(filtered_records)

    print(f"Positive file created with {len(filtered_records)} records")

    return number_positive
