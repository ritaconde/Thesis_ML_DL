from Bio import Entrez
from Bio import SeqIO
import numpy as np
import re

def create_related_domains(data_row):

    print("Number of related domains (PFAM & CDD)")

    related_domains = 0
    uniprot_id = data_row[1]

    try:
        Entrez.email = "example@example.com"

        handle = Entrez.efetch(db="protein", id=uniprot_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, format="genbank")
        annotations = record.annotations
        db_source = annotations.get("db_source")

        if db_source != None:
            Pfams= re.findall("(Pfam:.+?),", db_source)
            Cdds= re.findall("(CDD:.+?),", db_source)
            
            related_domains = len(Pfams) + len(Cdds)

    except Exception as e:
        related_domains = 0
    
    result = np.array(related_domains)
    result.astype(np.int64)

    return result

def get_related_domains_columns():
    
    return ["#RelatedDomains"]