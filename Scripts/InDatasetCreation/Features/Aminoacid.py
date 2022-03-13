import numpy as np

aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    #        0                                                                           19 

columns = ["Alanine-A","Arginine-R","Asparagine-N","Aspartiv Acid-D","Cysteine-C","Glumanine-Q",
                       "Glutamic acid-E","Glycine-G","Histidine-H","Isoleucine-I","Leucine-L","Lysine-K",
                       "Methionine-M","Phenylalanine-F","Proline-P","Serine-S","Threonine-T","Tryptophan-W",
                       "Tyrosine-Y","Valine-V"]

chemical_columns = ["Charged(DEKHR)","Aliphatic(ILV)","Aromatic(FHWY)","Polar(DERKQN)","Neutral(AGHPSTY)","Hydrophobic(CFILMVW)","+charged(KRH)","-charged(DE)","Tiny(ACDGST)","Small(EHILKMNPQV)","Large(FRWY)"]

def count_aminoacid(aminoacid,sequence):
    """
    Returns the number of a given aminoacid in a sequence
    """

    return sequence.count(aminoacid)

def create_aminoacid_composition(data):
    """
    Returns an array with the aminoacid composition of each example in the dataset
    """

    print("Aminoacid composition")

    sequence = data[2]
    sequence_size = len(sequence)

    result = np.array((count_aminoacid(aminoacids[0],sequence)/sequence_size,
                        count_aminoacid(aminoacids[1],sequence)/sequence_size,
                        count_aminoacid(aminoacids[2],sequence)/sequence_size,
                        count_aminoacid(aminoacids[3],sequence)/sequence_size,
                        count_aminoacid(aminoacids[4],sequence)/sequence_size,
                        count_aminoacid(aminoacids[5],sequence)/sequence_size,
                        count_aminoacid(aminoacids[6],sequence)/sequence_size,
                        count_aminoacid(aminoacids[7],sequence)/sequence_size,
                        count_aminoacid(aminoacids[8],sequence)/sequence_size,
                        count_aminoacid(aminoacids[9],sequence)/sequence_size,
                        count_aminoacid(aminoacids[10],sequence)/sequence_size,
                        count_aminoacid(aminoacids[11],sequence)/sequence_size,
                        count_aminoacid(aminoacids[12],sequence)/sequence_size,
                        count_aminoacid(aminoacids[13],sequence)/sequence_size,
                        count_aminoacid(aminoacids[14],sequence)/sequence_size,
                        count_aminoacid(aminoacids[15],sequence)/sequence_size,
                        count_aminoacid(aminoacids[16],sequence)/sequence_size,
                        count_aminoacid(aminoacids[17],sequence)/sequence_size,
                        count_aminoacid(aminoacids[18],sequence)/sequence_size,
                        count_aminoacid(aminoacids[19],sequence)/sequence_size))

    result.astype(np.float64)

    return result

def create_aminoacid_occurrence(data):
    """
    Returns an array with the aminoacid occurence of each example in the dataset
    """

    print("Aminoacid occurence")

    sequence = data[2]
    
    result = np.array((count_aminoacid(aminoacids[0],sequence),count_aminoacid(aminoacids[1],sequence),
                          count_aminoacid(aminoacids[2],sequence),count_aminoacid(aminoacids[3],sequence),
                          count_aminoacid(aminoacids[4],sequence),count_aminoacid(aminoacids[5],sequence),
                          count_aminoacid(aminoacids[6],sequence),count_aminoacid(aminoacids[7],sequence),
                          count_aminoacid(aminoacids[8],sequence),count_aminoacid(aminoacids[9],sequence),
                          count_aminoacid(aminoacids[10],sequence),count_aminoacid(aminoacids[11],sequence),
                          count_aminoacid(aminoacids[12],sequence),count_aminoacid(aminoacids[13],sequence),
                          count_aminoacid(aminoacids[14],sequence),count_aminoacid(aminoacids[15],sequence),
                          count_aminoacid(aminoacids[16],sequence),count_aminoacid(aminoacids[17],sequence),
                          count_aminoacid(aminoacids[18],sequence),count_aminoacid(aminoacids[19],sequence)))
    
    result.astype(np.int64)

    return result

def create_aminoacid_physico_chemical_occurrence(data):
    """
    Returns an array with the aminoacid composition based on the physico-chemical properties of each example in the dataset
    """

    print("Aminoacid physico chemical occurence")
    
    sequence = data[2]

    result = np.array((count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[1], sequence),
                        count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[19], sequence),
                        count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[17], sequence)+count_aminoacid(aminoacids[18], sequence),
                        count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[5], sequence)+count_aminoacid(aminoacids[2], sequence),
                        count_aminoacid(aminoacids[0], sequence)+count_aminoacid(aminoacids[7], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[14], sequence)+count_aminoacid(aminoacids[15], sequence)+count_aminoacid(aminoacids[16], sequence)+count_aminoacid(aminoacids[18], sequence),
                        count_aminoacid(aminoacids[4], sequence)+count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[12], sequence)+count_aminoacid(aminoacids[19], sequence)+count_aminoacid(aminoacids[17], sequence),
                        count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[8], sequence),
                        count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence),
                        count_aminoacid(aminoacids[0], sequence)+count_aminoacid(aminoacids[4], sequence)+count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[7], sequence)+count_aminoacid(aminoacids[15], sequence)+count_aminoacid(aminoacids[16], sequence),
                        count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[12], sequence)+count_aminoacid(aminoacids[2], sequence)+count_aminoacid(aminoacids[14], sequence)+count_aminoacid(aminoacids[5], sequence)+count_aminoacid(aminoacids[19], sequence),
                        count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[17], sequence)+count_aminoacid(aminoacids[18], sequence)))

    result.astype(np.int64)

    return result

def create_aminoacid_physico_chemical_composition(data): 
    """
    Returns an array with the aminoacid composition based on the physico-chemical properties of each example in the dataset
    """
    
    print("Aminoacid physico chemical composition")

    sequence = data[2]

    result = np.array(((count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[1], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[19], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[17], sequence)+count_aminoacid(aminoacids[18], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[5], sequence)+count_aminoacid(aminoacids[2], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[0], sequence)+count_aminoacid(aminoacids[7], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[14], sequence)+count_aminoacid(aminoacids[15], sequence)+count_aminoacid(aminoacids[16], sequence)+count_aminoacid(aminoacids[18], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[4], sequence)+count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[12], sequence)+count_aminoacid(aminoacids[19], sequence)+count_aminoacid(aminoacids[17], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[8], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[6], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[0], sequence)+count_aminoacid(aminoacids[4], sequence)+count_aminoacid(aminoacids[3], sequence)+count_aminoacid(aminoacids[7], sequence)+count_aminoacid(aminoacids[15], sequence)+count_aminoacid(aminoacids[16], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[6], sequence)+count_aminoacid(aminoacids[8], sequence)+count_aminoacid(aminoacids[9], sequence)+count_aminoacid(aminoacids[10], sequence)+count_aminoacid(aminoacids[11], sequence)+count_aminoacid(aminoacids[12], sequence)+count_aminoacid(aminoacids[2], sequence)+count_aminoacid(aminoacids[14], sequence)+count_aminoacid(aminoacids[5], sequence)+count_aminoacid(aminoacids[19], sequence))/len(sequence),
                          (count_aminoacid(aminoacids[13], sequence)+count_aminoacid(aminoacids[1], sequence)+count_aminoacid(aminoacids[17], sequence)+count_aminoacid(aminoacids[18], sequence))/len(sequence)))

    result.astype(np.float64)

    return result

def dipeptide_composition(sequence):
    dipeptides = get_dipeptide_composition_columns()
    seq_size = len(sequence)-1
    dipcompo = []

    for dip in dipeptides:

        count = sequence.count(dip)

        dipcomposition = count/seq_size
        dipcompo.append(dipcomposition)
    
    return tuple(dipcompo)

def create_dipeptide_composition(data): 
    """
    Returns an array with the dipeptide composition of each example in the dataset
    """

    print("Aminoacid dipeptide composition")

    sequence = data[2]

    result = np.array(dipeptide_composition(sequence))

    result.astype(np.float64)

    return result

def get_aminoacid_occurence_columns():

    return ["Aminoacid_Occurence:" + s for s in columns]

def get_aminoacid_composition_columns():

    return ["Aminoacid_Composition:" + s for s in columns]

def get_aminoacid_physico_chemical_occurrence_columns():

    return ["Aminoacid_Physico_Occurence:" + s for s in chemical_columns]

def get_aminoacid_physico_chemical_composition_columns():

    return ["Aminoacid_Physico_Composition:" + s for s in chemical_columns]

def get_dipeptide_composition_columns():

    dipeptides=[]
    for item in aminoacids:
        pep1=item
        for item in aminoacids:
            pep2=str(pep1)+str(item)
            dipeptides.append(pep2)

    return dipeptides