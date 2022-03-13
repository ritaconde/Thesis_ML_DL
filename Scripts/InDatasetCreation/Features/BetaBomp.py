import requests
import numpy as np
import re
import time

BOMP_START_URL = "http://services.cbu.uib.no/tools/bomp/handleForm"
BOMP_RESULT_URL = "http://services.cbu.uib.no/tools/bomp/viewOutput"

def get_bomp_job_id(sequence):

    params = {'seqs': sequence, 'useblast': "on", 'evalue': 0.0000000001, 'queryfile': "", 'SUBMIT': "Submit Search"}
    
    bomp_start_result = requests.post(BOMP_START_URL, data = params)

    if not bomp_start_result.ok:
        print("BOMP Start Process Error")
        print(bomp_start_result.content)
        raise Exception("BOMP Start Process Error")
    
    request_content = str(bomp_start_result.content).strip("b'")

    aux = re.search("viewOutput\?id=(.*?)\"", request_content)
    job_id = aux.group(1)

    return job_id

def get_bomp_job_result(job_id):

    params = {'id': job_id }

    while True:
        time.sleep(5)

        phobius_job_result = requests.get(BOMP_RESULT_URL, params=params)
        request_content = str(phobius_job_result.content).strip("b'")

        regex = re.search("barrel outer membrane proteins predicted is:  (.*?)<", request_content)

        if regex != None:
            result = regex.group(1)

            return int(result)


def create_betabarrels(data_row):
    
    print("Count number beta barrels (BOMP)")

    sequence = data_row[2]

    bomp_job_id = get_bomp_job_id(sequence)
    bomp_result = get_bomp_job_result(bomp_job_id)

    result = np.array(bomp_result)
    result.astype(np.int64)

    return result

def get_beta_columns():

    return ["#BompBetaBarrels"]