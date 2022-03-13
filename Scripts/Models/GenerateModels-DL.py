import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from propythia.deep_ml import DeepML
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from Bio import SeqIO


aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def get_base_data():
    # Load base data
    base_data = pd.read_csv('/Users/marcelo/Desktop/Rita/Tese/tese.nosync/Data/data-filtered-tcdb89.csv', sep=',')
    mask = (base_data['Sequence'].str.len() < 1500)

    transporters = base_data.loc[base_data["Is transporter?"] == 1].loc[mask]
    non_transporters = (base_data.loc[base_data["Is transporter?"] == 0].loc[mask]).sample(transporters.shape[0] * 2)
    # non_transporters = (base_data.loc[base_data["Is transporter?"] == 0].loc[mask]).sample(transporters.shape[0])

    frames = pd.concat([transporters, non_transporters])

    return frames

# Calculate longest sequence length
def calculate_longest_sequence(sequences):
    max_len = 0

    for seq in sequences:
        leng = len(seq)
        if max_len < leng:
            max_len = leng

    return max_len

# Calculate shortest sequence length
def calculate_shortest_sequence(sequences):
    min_len = len(sequences[0])

    for seq in sequences:
        leng = len(seq)
        if min_len > leng:
            min_len = leng

    return min_len

# Create a dictionary for a list based on index
def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict


def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
    """
    char_dict = create_dict(aminoacids)

    encode_list = []
    for row in data:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        
        encode_list.append(np.array(row_encode))

    return encode_list

def encode_transporter_class(tr_data):
    encoder = LabelEncoder()
    encoder.fit(tr_data)
    fps_y_encoded = encoder.transform(tr_data)

    return fps_y_encoded
    # total_sequences_length = 0
    # sequences_size = []
    # for se in input_data:
    #     se_size = len(se)
    #     sequences_size.append(se_size)
    #     total_sequences_length += se_size

    # min_len = calculate_shortest_sequence(input_data)
    # medium_size = total_sequences_length / total_sequences
    # total_sequences = input_data.shape[0]

    # print(f"Medium: {medium_size}")
    # print(f"Max length: {max_len}")
    # print(f"Min length: {min_len}")

    # fig = plt.figure(figsize =(10, 7))
    # # Creating plot
    # plt.boxplot(sequences_size)
    
    # # show plot
    # plt.savefig("example.png", dpi=150, transparent=True)
    # plt.show()
    # exit()

# Function that pre processes a list of sequences based on a list of amino acids
def pre_process_dataset(input_data, max_len = 0):

    if max_len == 0:
        max_len = calculate_longest_sequence(input_data)

    # encode each amino acid
    enconded_data = integer_encoding(input_data)

    # normalize sequences with the same size - max sequence length
    enconded_data_pad = pad_sequences(enconded_data, maxlen=max_len, padding='post')

    # one hot encoded vectorization
    hot2d = to_categorical(enconded_data_pad)

    fps_x_hot1d = hot2d.reshape(hot2d.shape[0], hot2d.shape[1]*hot2d.shape[2])

    return fps_x_hot1d

def generate_cnn2d(input_data, output_data, predict = None):
    max_len=calculate_longest_sequence(input_data)
    # Preprocess dataset
    input_data_preprocessed = pre_process_dataset(input_data, max_len= max_len)

    # Encode output data
    predict_data = encode_transporter_class(output_data)
    # input_dim = (calculate_longest_sequence(input_data), len(aminoacids), 1)
    input_dim = input_data_preprocessed.shape[1]

    # Split on train, test a validation sets
    x_train, x_res, y_train, y_res = train_test_split(input_data_preprocessed, predict_data, train_size=0.8)
    x_valid, x_test, y_valid, y_test = train_test_split(x_res,y_res, test_size=0.5)

    x_train, x_test, x_valid = map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [x_train, x_test, x_valid])

    dl = DeepML(x_train, y_train, x_test, y_test,
                x_dval=x_valid, y_dval=y_valid,
                tensorboard = True,
                epochs=500, batch_size=128, verbose = 1, report_name = "DL-cnn/results")


    if predict is not None:
        input_data_preprocessed = pre_process_dataset(predict, max_len)
        input = input_data_preprocessed.reshape(input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
        model = dl.load_model('DL-cnn-best/final.h5')
        prediction = model.predict(input)

        flat_list = [item for sublist in prediction for item in sublist]

        return np.array(flat_list)

    else:
        dl.run_cnn_1D(
            input_dim=input_dim,
            cv=None,
            filter_count=(128, 256, 256, 128),
            dense_layers = (128, 256, 256, 128),
            dropout_rate=(0.5,0.5,0.5,0.5),
            dropout_cnn = (0.5,0.5,0.5,0.5),
            l1=0.00001, l2=0.0001,
            scoring=make_scorer(matthews_corrcoef))

        dl.model_complete_evaluate()

        dl.save_model(path="DL-cnn/final.h5")

def generate_lstm(input_data, output_data, predict = None):
    max_len=calculate_longest_sequence(input_data)

    # Preprocess dataset
    input_data_preprocessed = pre_process_dataset(input_data, max_len)

    # Encode output data
    predict_data = encode_transporter_class(output_data)
    # input_dim = (calculate_longest_sequence(input_data), len(aminoacids), 1)
    input_dim = input_data_preprocessed.shape[1]

    # Split on train, test a validation sets
    x_train, x_res, y_train, y_res = train_test_split(input_data_preprocessed, predict_data, train_size=0.8)
    x_valid, x_test, y_valid, y_test = train_test_split(x_res,y_res, test_size=0.5)

    x_train, x_test, x_valid = map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [x_train, x_test, x_valid])

    dl = DeepML(x_train, y_train, x_test, y_test,
                x_dval=x_valid, y_dval=y_valid,
                epochs=100, batch_size=128, verbose = 1, report_name = "DL-lstm/results")

    if predict is not None:
        input_data_preprocessed = pre_process_dataset(predict, max_len)
        input = input_data_preprocessed.reshape(input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
        model = dl.load_model('DL-lstm-best/final.h5')

        prediction = model.predict(input)

        flat_list = [item for sublist in prediction for item in sublist]

        return np.array(flat_list)

    else:
        dl.run_lstm_simple(
            input_dim=input_dim,
            cv=None,
            lstm_layers=(256, 128, 128, 64),
            dense_layers=(256, 128, 128,64),
            dropout_rate=(0.5,0.5,0.5,0.5), recurrent_dropout_rate=(0.5,0.5,0.5,0.5),
            dropout_rate_dense=(0.5,0.5,0.5,0.5),
            l1=0.00001, l2=0.0001,
            scoring=make_scorer(matthews_corrcoef))

        dl.model_complete_evaluate()

        dl.save_model(path="DL-lstm/final.h5")

def generate_hybrid(input_data, output_data, predict = None):
    max_len=calculate_longest_sequence(input_data)

    # Preprocess dataset
    input_data_preprocessed = pre_process_dataset(input_data, max_len)

    # Encode output data
    predict_data = encode_transporter_class(output_data)
    # input_dim = (calculate_longest_sequence(input_data), len(aminoacids), 1)
    input_dim = input_data_preprocessed.shape[1]

    print(input_data_preprocessed.shape)
    exit()

    # Split on train, test a validation sets
    x_train, x_res, y_train, y_res = train_test_split(input_data_preprocessed, predict_data, train_size=0.8)
    x_valid, x_test, y_valid, y_test = train_test_split(x_res,y_res, test_size=0.5)

    x_train, x_test, x_valid = map(lambda x: x.reshape(x.shape[0], 1, x.shape[1]), [x_train, x_test, x_valid])

    dl = DeepML(x_train, y_train, x_test, y_test,
                x_dval=x_valid, y_dval=y_valid,
                epochs=100, batch_size=128, verbose = 1, report_name = "Hybrid-best/results")

    if predict is not None:
        input_data_preprocessed = pre_process_dataset(predict, max_len)
        input = input_data_preprocessed.reshape(input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
        model = dl.load_model('Hybrid-best/final.h5')

        prediction = model.predict(input)

        flat_list = [item for sublist in prediction for item in sublist]

        return np.array(flat_list)

    else:
            
        dl.run_cnn_lstm(
                    input_dim,
                    optimizer='Adam',
                    filter_count=(256, 128),
                    padding='same',
                    strides=1,
                    kernel_size=(3,),
                    cnn_activation=None,
                    kernel_initializer='glorot_uniform',
                    dropout_cnn = (0.5,0.5),
                    max_pooling=(True,),
                    pool_size=(2,), strides_pool=1,
                    data_format_pool='channels_first',
                    bilstm=True,
                    lstm_layers=(128,64),
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    dropout_rate=(0.5,0.5),
                    recurrent_dropout_rate=(0.5,0.5),
                    l1=1e-5, l2=1e-4,
                    dense_layers = (256, 128),
                    dense_activation="relu",
                    dropout_rate_dense=(0.5,0.5),
                    batchnormalization=(True,),
                    loss_fun = None, activation_fun = None,
                    cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
                    scoring=make_scorer(matthews_corrcoef))

        dl.model_complete_evaluate()

        dl.save_model(path="Hybrid/final.h5")

def get_data_to_predict():
    fasta_sequences = list(SeqIO.parse('../Cere/cere.faa', 'fasta'))
    res = []
    seq_ids = []

    for sequence in fasta_sequences:
        res.append(str(sequence.seq))
        seq_ids.append(sequence.id)
    
    return (np.array(res), np.array(seq_ids))

def main():

    (data_to_predict, data_to_predict_ids) = get_data_to_predict()

    # Get data referent to current filtered data with
    base_data = get_base_data()

    # Mix dataset
    shuffle_data = base_data.sample(frac=1)

    # Split in input and output atributes
    input_data = np.array(shuffle_data["Sequence"])
    output_data = np.array(shuffle_data["Is transporter?"])

    kybrid_predict = generate_hybrid(input_data, output_data, data_to_predict)

    # cnn2d_predict = generate_cnn2d(input_data, output_data, data_to_predict)
    # # lstm_predict = generate_lstm(input_data, output_data, data_to_predict)

    con = np.stack((data_to_predict_ids, kybrid_predict), axis=1)

    np.savetxt("../Cere/cere-hybrid.csv", con, fmt="%s;%s")

if __name__ == '__main__':
    main()
