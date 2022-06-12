


import load_data
import torch
import json,argparse
import load_model
import roberta_training
import corenlp_local
import seq2sql_model_testing
import seq2sql_model_training_functions
import model_save_and_infer
import dev_function
import infer_functions
import time
import os
import nltk

from dbengine_sqlnet import DBEngine
from torchsummary import summary
from tqdm.notebook import tqdm
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings("ignore")

# The following cell will set the PyTorch device to a GPU which enables us to use it during runtime.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# Loading Data From Files
path_wikisql = "C:\\Users\\Kasun\\PycharmProjects\\corps\\NL2SQL"
BATCH_SIZE = 8

train_data, train_table, dev_data, dev_table, train_loader, dev_loader = load_data.get_data(path_wikisql, batch_size = BATCH_SIZE)
test_data,test_table,test_loader = load_data.get_test_data(path_wikisql, batch_size = BATCH_SIZE)
zero_data,zero_table,zero_loader = load_data.get_zero_data(path_wikisql, batch_size = BATCH_SIZE)    # Data to test Zero Shot Learning


# Loading Models
roberta_model, tokenizer, configuration = load_model.get_roberta_model()          # Loads the RoBERTa Model
seq2sql_model = load_model.get_seq2sql_model(configuration.hidden_size)           # Loads the LSTM based submodels



# Loading the Pre trained weights, skip the below cell if you want to train the model from scratch

path_roberta_pretrained = path_wikisql + "/model_roberta_best.pt"
path_model_pretrained = path_wikisql + "/model_best.pt"

if torch.cuda.is_available():
    res = torch.load(path_roberta_pretrained)
else:
    res = torch.load(path_roberta_pretrained, map_location='cpu')

roberta_model.load_state_dict(res['model_roberta'])

if torch.cuda.is_available():
    res = torch.load(path_model_pretrained)
else:
    res = torch.load(path_model_pretrained, map_location='cpu')

seq2sql_model.load_state_dict(res['model'])



# Loading the Model Optimizers
# RoBERTa: Adam Optimizer with learning rate = 0.00001
# SubModels: Adam Optimizer with learning rate = 0.001
model_optimizer, roberta_optimizer = load_model.get_optimizers(seq2sql_model , roberta_model)


# Below we define a function that prints the metrics in a readable format

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


# Training the Model

EPOCHS = 5

acc_lx_t_best = 0.693  # Creats checkpoint so that a worse model does not get saved
epoch_best = 0
for epoch in range(EPOCHS):
    acc_train = dev_function.train(seq2sql_model, roberta_model, model_optimizer, roberta_optimizer, tokenizer,
                                   configuration, path_wikisql, train_loader)
    acc_dev, results_dev, cnt_list = dev_function.test(seq2sql_model, roberta_model, model_optimizer, tokenizer,
                                                       configuration, path_wikisql, dev_loader, mode="dev")
    print_result(epoch, acc_train, 'train')
    print_result(epoch, acc_dev, 'dev')
    acc_lx_t = acc_dev[-2]
    if acc_lx_t > acc_lx_t_best:  # IMPORTANT : Comment out this whole if block if you are using a shortcut to the original
        acc_lx_t_best = acc_lx_t  # Drive Folder, otherwise an error will stop the execution of the code.
        epoch_best = epoch  # You cannot edit the files in the original folder
        #             Download and Upload a separate copy to change the files.

        # save best model
        state = {'model': seq2sql_model.state_dict()}
        torch.save(state, os.path.join(path_wikisql, 'model_best.pt'))

        state = {'model_roberta': roberta_model.state_dict()}
        torch.save(state, os.path.join(path_wikisql, 'model_roberta_best.pt'))

    print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")


# Testing The Model

# acc_dev, results_dev, _ = dev_function.test(seq2sql_model, roberta_model, model_optimizer, tokenizer, configuration, path_wikisql, dev_loader, mode="dev")
# acc_test, results_test, _ = dev_function.test(seq2sql_model, roberta_model, model_optimizer, tokenizer, configuration, path_wikisql, test_loader, mode="test")
# acc_zero, results_zero, _ = dev_function.test(seq2sql_model, roberta_model, model_optimizer, tokenizer, configuration, path_wikisql, zero_loader, mode="test")
#
# print_result('test', acc_dev, 'dev')
# print_result('test', acc_test, 'test')
# print_result('test', acc_zero, 'zero')

# Test You Own Queries!

nlu = "Which year did the band release the Song 'Wake me Up'?"

# Specify the Table Schema
table_id = '1-10015132-16'
headers = ['Band', 'Song', 'Studio', 'Year', 'Awards']
types = ['text', 'text', 'text', 'text', 'text']

pr_sql_i =  infer_functions.infer(
                nlu,
                table_id, headers, types, tokenizer,
                seq2sql_model, roberta_model, configuration, max_seq_length=222,
                num_target_layers=2,
                beam_size=4
            )
print(pr_sql_i)