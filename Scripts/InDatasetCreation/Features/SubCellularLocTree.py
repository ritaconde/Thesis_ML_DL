from bs4 import BeautifulSoup
import numpy as np
import requests
import time
import uuid
import json

LOCTREE_START_URL = "https://rostlab.org/services/loctree3/submit-job.php" # POST
LOCTREE_STATUS_URL = "https://rostlab.org/~loctree3/qjobstat.php" # POST
LOCTREE_RESULT_URL = "https://rostlab.org/services/loctree3/results.php" # GET

DONE_STATUS = "done"


def get_sequence_domain_string(domain):

    dic = {"Eukaryota": "euka", "Bacteria": "bact", "Archaea": "arch"}

    if domain in dic:
        return dic[domain]

    else:
        return dic["Eukaryota"]


def get_loctree_job_id(sequence, domain, sequence_id):
    i = 0

    list_domains = ["Eukaryota", "Bacteria", "Archaea"]

    while True:
        id = str(uuid.uuid4())
        sequence = f">{sequence_id}\n" + sequence

        dom_str = get_sequence_domain_string(domain)

        params = {'domain': dom_str, 'sequence': sequence, 'md5': id}

        loctree_start_result = requests.post(LOCTREE_START_URL, data = params)

        if not loctree_start_result.ok:
            print("LOCTREE Start Process Error")
            print(loctree_start_result.content)
            raise Exception("LOCTREE Status Process Error")

        json_result = str(loctree_start_result.content).strip("b'")
        json_data = json.loads(json_result)

        reqid = json_data["reqid"]

        status_result = check_job_status(reqid, domain)

        if status_result == DONE_STATUS:
            return id
        elif i == 2:
            return None

        list_domains.remove(domain)
        domain = list_domains[0]
        i += 1
            
def check_job_status(req_id, domain):
    params = {'jid': req_id}
    i = 0

    status = "err"
    while True:
        print(f"Iter {i} for domain {domain}")
        time.sleep(20)

        loctree_status_result = requests.post(LOCTREE_STATUS_URL, data = params)

        if not loctree_status_result.ok:
            print("LOCTREE Status Process Error")
            print(loctree_status_result.content)
            raise Exception("LOCTREE Status Process Error")
        
        json_result = str(loctree_status_result.content).strip("b'")
        json_data = json.loads(json_result)
        job_status = json_data["status"]

        if job_status == "unknown" or i > 12:
            status = DONE_STATUS
            break

        if job_status != "running" and job_status != "waiting":
            print(job_status)

        i+=1

    return status

def get_loctree_result_job_id(md5):

    params = {'id': md5}

    loctree_result = requests.get(LOCTREE_RESULT_URL, params = params)
    
    if not loctree_result.ok:
        print("LOCTREE Result Process Error")
        print(loctree_result.content)
        raise Exception("LOCTREE Result Process Error")

    job_result = str(loctree_result.content).strip("b'")

    soup = BeautifulSoup(job_result, features="html.parser")

    table_body = soup.find("div", attrs={'role':'table_body'})

    if table_body is None:
        return 0

    location = ""
    i=0
    table_rows = table_body.find_all("div", attrs={'role':'table_cell'})

    for row in table_rows:
        if i == 4:
            location = row.text.strip()
            break
        i+=1
    
    loc_id = get_location_id(location.lower())

    return loc_id

def get_location_id(location):

    dictionary = {
        "chloroplast": 1,
        "chloroplast membrane": 2,
        "cytosol": 3,
        "endoplasmic reticulum" : 4,
        "endoplasmic reticulum membrane": 5,
        "extra-cellular": 6,
        "ﬁmbrium": 7,
        "golgi apparatus": 8,
        "golgi apparatus membrane": 9,
        "mitochondrion": 10,
        "mitochondrion membrane": 11,
        "nucleus": 12,
        "nucleus membrane": 13,
        "outer membrane": 14,
        "periplasmic space": 15,
        "peroxisome": 16,
        "peroxisome membrane": 17,
        "plasma membrane": 18,
        "plastid": 19,
        "vacuole": 20,
        "vacuole membrane": 21,
        "secreted": 21,
        "cytoplasm": 23,
        "inner membrane": 24
    }

    if location in dictionary:
        return dictionary[location]

    return 0

def create_subcellular_location(data_row):

    print("Protein Subcellular Location (LocTree)")

    sequence_id = data_row[0]
    sequence = data_row[2]
    domain = data_row[5]

    loctree_job = get_loctree_job_id(sequence, domain, sequence_id)

    loctree_result = 0
    if loctree_job != None:
        loctree_result = get_loctree_result_job_id(loctree_job)

    result = np.array(loctree_result)
    result.astype(np.int64)

    return result

def get_subcellular_columns():

    return ["SubcellularLoc"]