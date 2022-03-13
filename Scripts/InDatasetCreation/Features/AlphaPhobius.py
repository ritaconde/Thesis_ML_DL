import numpy as np
import requests
import time

PHOBIOS_START_URL = "http://www.ebi.ac.uk/Tools/services/rest/phobius/run/"
PHOBIUS_STATUS_URL = "http://www.ebi.ac.uk/Tools/services/rest/phobius/status/{0}"
PHOBIOS_RESULT_URL = "http://www.ebi.ac.uk/Tools/services/rest/phobius/result/{0}/{1}"

def get_phobius_job_id(sequence):

    parameters = {'email': "example1234uminho@example.com", 'format': "short", 'stype': "protein", 'sequence': sequence}

    phobius_start_result = requests.post(PHOBIOS_START_URL, data = parameters)

    if not phobius_start_result.ok:
        print("Phobius Start Process Error")
        print(phobius_start_result.content)
        raise Exception("Phobius Start Process Error")

    content = phobius_start_result.content

    result = str(content).strip("b'")

    return result

def get_phobius_job_result(job_id):

    while True:
        time.sleep(5)

        phobius_job_status = requests.get(PHOBIUS_STATUS_URL.format(job_id))

        result_status = str(phobius_job_status.content).strip("b'")

        if result_status == "FINISHED":
            break

        if result_status != "RUNNING":
            print("Phobius JOB Error")
            raise Exception("Phobius JOB Error")

    phobius_job_result = requests.get(PHOBIOS_RESULT_URL.format(job_id, "out"))

    if not phobius_job_result.ok:
        print("Phobius Job Result Error")
        print(phobius_job_result.content)
        raise Exception("Phobius Job Result Error")
    
    content = phobius_job_result.content
    data = content.split(b"\n")[1].split()
    
    number_alpha_helices_str = str(data[1]).strip("b'")

    number_alpha_helices = 0

    if number_alpha_helices_str.isnumeric():
        number_alpha_helices = int(number_alpha_helices_str)

    signal_peptide = 0

    signal = str(data[2]).strip("b'")

    if signal == "Y" or signal == "found":
        signal_peptide = 1

    return (number_alpha_helices, signal_peptide)

def create_num_alphahelices_and_signalpeptide(data_row):
    ''' Returns an array with number of alpha helices and the estimation result of signal peptides with Phobius tool '''

    print("Count number alpha helices and signal peptide (Phobius)")

    sequence = data_row[2]

    phobius_job_id = get_phobius_job_id(sequence)
    phobius_result = get_phobius_job_result(phobius_job_id)

    result = np.array(phobius_result)
    result.astype(np.int64)

    return result

def get_alpha_columns():

    return ["#PhobiusAlfa", "PhobiusSignalPeptide"]
