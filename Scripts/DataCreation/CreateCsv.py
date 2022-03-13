import DataCreation.Constants as Constants
from Bio import SeqIO
import numpy as np
import pandas as pd
import Utils.File as File
import Utils.String as String
from Bio import Entrez

def __get_sequence_id(record):

    return str(record.id)

def __get_sequence(record):

    return str(record.seq)

def __get_uniprot_accession(record):
    y=record.id[0:3]
    IDs=record.id.split("|")
    if y == "gnl": #positive case
        ID=IDs[2]
    elif len(IDs) > 1: #negative case
        ID=IDs[1]
    else:
        ID=IDs[0]

    return ID

def __get_tcdb_id(record):
    seq_id = __get_sequence_id(record)
    res = ""
    i = len(seq_id)-1
    while seq_id[i] != "|":
        res += seq_id[i]
        i-=1
    return str(String.reverse(res))

def __get_domain(record):
    uniprot_asc = __get_uniprot_accession(record)

    try:
        Entrez.email = "example@example.com"

        handle = Entrez.efetch(db="protein", id=uniprot_asc, rettype="gb", retmode="text")

        record = SeqIO.read(handle, format="genbank")
        annot = record.annotations

        taxonomy = annot.get("taxonomy")

        domain = "Unknown"

        if len(taxonomy) > 0:
            domain = taxonomy[0]

        return domain

    except Exception as error:

        print(f"Error trying to fetch domain for sequence {uniprot_asc}: {error}")

        return '0'

def __populate_positive_data(dataset):

    # Get positive file path
    paths = Constants.POSITIVE_PATH.copy()
    paths.append(Constants.POSITIVE_FILTERED_DATA_FILE)
    positive_file_path = File.get_file_path(paths)

    # Load positive fasta file
    handler = open(positive_file_path, "rU")
    records = list(SeqIO.parse(handler, "fasta"))
    handler.close()

    lastperc = 0
    i = 1
    total = len(records)
    # Iterate over positive records and add them to dataset numpy array
    for record in records:
        data=np.array((__get_sequence_id(record), __get_uniprot_accession(record), __get_sequence(record), '1' , __get_tcdb_id(record), __get_domain(record)), dtype=object)
        dataset=np.vstack((dataset,data))
        
        perc = int(i/total * 100)

        if perc > lastperc:
            print(f"Positive populate: {perc}%")

        lastperc = perc
        i+=1

    return dataset

def __populate_negative_data(dataset):

    # Get negative file path
    paths = Constants.NEGATIVE_PATH.copy()
    paths.append(Constants.NEGATIVE_FILTERED_DATA_FILE)
    negative_file_path = File.get_file_path(paths)

    # Load negative fasta file
    handler = open(negative_file_path, "rU")
    records = list(SeqIO.parse(handler, "fasta"))
    handler.close()

    lastperc = 0
    i = 1
    total = len(records)
    # Iterate over negative records and add them to dataset numpy array
    for record in records:
        data=get_default_record_data(record)
        dataset=np.vstack((dataset,data))

        perc = int(i/total * 100)

        if perc > lastperc:
            print(f"Negative populate: {perc}%")

        lastperc = perc
        i+=1

    return dataset


def get_default_record_data(record):

    return np.array((__get_sequence_id(record), __get_uniprot_accession(record), __get_sequence(record), '0' , '0', __get_domain(record)), dtype=object)

def generate_data_csv():
    
    # Create dataset pointer
    dataset=np.array(('Fasta ID','Uniprot Accession', 'Sequence', 'Is transporter?','TCDB ID','Taxonomy Domain'), dtype=object)

    print("Start populate positive data")
    # Populate dataset pointer with positive data
    dataset = __populate_positive_data(dataset)

    print("Start populate negative data")

    # Populate dataset pointer with negative data
    dataset = __populate_negative_data(dataset)

    # Print on file dataset
    paths = Constants.MAIN_DATA_PATH.copy()
    paths.append(Constants.MAIN_DATA_FILE)
    data_file_path = File.get_file_path(paths)

    dataframe = pd.DataFrame(dataset[1:], index=None, columns=dataset[0])
    dataframe.to_csv(data_file_path, sep=',', index=False)

    print("Data filtered csv file created")
