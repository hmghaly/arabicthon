

ar_chars=[chr(i) for i in range(1560,1630)]
ar_chars+=" "

#تحويل المسميات غير الرقمية (الفئات) إلى متجه صفري ليس به إلا واحد فقط مناظر لترتيب العنصر الحالي ضمن قائمة الفئات
def one_hot_encoder(item,item_list):
  zeros=[0.]*len(item_list)
  if item in item_list:
    item_index=item_list.index(item)
    zeros[item_index]=1.
  return zeros
#استخدام سمات الحروف التي تبدأ بها وتنتهي بها الكلمة لعمل متجه من الأرقام بالسمات المميزة للكلمة سيتم تغذية الشبكة العصبية بها
def extract_word_feature_NEW(word0,n_chars=3):
  feature_list=[]
  flat_feature_list=[]
  for i in range(n_chars):
    cur_char=" "
    if i<len(word0): cur_char=word0[i]
    cur_one_hot=one_hot_encoder(cur_char,ar_chars)
    flat_feature_list.extend(cur_one_hot)
  for i in range(n_chars):
    word1 =word0[-n_chars:] 
    cur_char=" "
    if i<len(word1): cur_char=word1[i]
    cur_one_hot=one_hot_encoder(cur_char,ar_chars)
    flat_feature_list.extend(cur_one_hot)
  return flat_feature_list


def extract_word_features(word0,params0={},additional_data=[]):
  n_chars=params0.get("n_chars",3)
  wv_model_fpath=params0.get("wv_model_path")
  wv_model=wv_dict[wv_model_fpath]
  feature_list=[]
  flat_feature_list=[]
  for i in range(n_chars):
    cur_char=" "
    if i<len(word0): cur_char=word0[i]
    cur_one_hot=one_hot_encoder(cur_char,ar_chars)
    flat_feature_list.extend(cur_one_hot)
  for i in range(n_chars):
    word1 =word0[-n_chars:] 
    cur_char=" "
    if i<len(word1): cur_char=word1[i]
    cur_one_hot=one_hot_encoder(cur_char,ar_chars)
    flat_feature_list.extend(cur_one_hot)
  if wv_model!=None:
    try:
      cur_vec=wv_model[word0]
    except:
      cur_vec=[0.]*wv_model.vector_size
    flat_feature_list.extend(cur_vec)
  flat_feature_list.extend(additional_data)

  return flat_feature_list

from collections import OrderedDict
#Let's build the network - here is a small cheat sheet for possible RNN classes based on input and output size
#https://github.com/hmghaly/rnn/blob/master/classes.py

#here the size of the output is the same as the size of the input
#the depth of the output depends on the number of possible outcome categories (e.g. different phonemes)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import dill as pickle
import numpy as np
#from code_utils import np_lstm

torch.manual_seed(1)
random.seed(1)

device = torch.device('cpu')
#device = torch.device('cuda')

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size,num_layers, matching_in_out=False, apply_sigmoid=False, apply_softmax=False, batch_size=1):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.apply_softmax=apply_softmax
    self.apply_sigmoid=apply_sigmoid
    self.matching_in_out = matching_in_out #length of input vector matches the length of output vector 
    self.lstm = nn.LSTM(input_size, hidden_size,num_layers)
    self.hidden2out = nn.Linear(hidden_size, output_size)
    if self.apply_softmax: self.softmax =nn.Softmax(dim=2)
    if self.apply_sigmoid: self.sigmoid =nn.Sigmoid() 
    
    #self.sigmoid = torch.sigmoid(dim=1)
    self.hidden = self.init_hidden()
  def forward(self, feature_list):
    self.hidden = self.init_hidden() ### check
    feature_list=torch.tensor(feature_list)
    feature_list=feature_list.to(device) #### <<<<<<<<<<<<<<<<< 
    if self.matching_in_out:
      lstm_out, _ = self.lstm( feature_list.view(len( feature_list), 1, -1))
      output_scores = self.hidden2out(lstm_out.view(len( feature_list), -1))
      if self.apply_sigmoid: output_scores=self.sigmoid(output_scores).to(device)
      elif self.apply_softmax: output_scores=self.softmax(output_scores).to(device)
      #output_scores = torch.sigmoid(output_space) #we'll need to check if we need this sigmoid
      return output_scores #output_scores
    else:
      outs=[]
      for i in range(len(feature_list)):
        cur_ft_tensor=feature_list[i]#.view([1,1,self.input_size])
        cur_ft_tensor=cur_ft_tensor.view([1,1,self.input_size])
        lstm_out, self.hidden = self.lstm(cur_ft_tensor, self.hidden)
        outs=self.hidden2out(lstm_out)
        if self.apply_sigmoid: outs = self.sigmoid(outs).to(device) #self.sigmoid =nn.Sigmoid()
        elif self.apply_softmax: outs = self.softmax(outs).to(device)
        
      return outs
  def init_hidden(self):
    #return torch.rand(self.num_layers, self.batch_size, self.hidden_size)
    return (torch.rand(self.num_layers, self.batch_size, self.hidden_size).to(device),
            torch.rand(self.num_layers, self.batch_size, self.hidden_size).to(device))


def out2labels(rnn_flat_out,label_list): #a flat rnn output to split into slices, and get the label weights for each slice
  final_list=[]
  n_slices=int(len(rnn_flat_out)/len(label_list))
  for i0 in range(n_slices):
    i1=i0+1
    cur_slice=rnn_flat_out[i0*len(label_list):i1*len(label_list)]
    tmp_list=[]
    for lb0,cs0 in zip(label_list,cur_slice): tmp_list.append((lb0,cs0))
    tmp_list.sort(key=lambda x:-x[-1])
    final_list.append(tmp_list)
  return final_list


class model_pred:
  def __init__(self,model_fpath0) -> None:
    # try: self.checkpoint = torch.load(model_fpath0)
    # except: self.checkpoint = dill_unpickle(model_fpath0)
    self.checkpoint = torch.load(model_fpath0)
    self.rnn = RNN(self.checkpoint["n_input"], self.checkpoint["n_hidden"] , self.checkpoint["n_output"] , self.checkpoint["n_layers"] , matching_in_out=self.checkpoint["n_layers"]).to(device)
    self.rnn.load_state_dict(self.checkpoint['model_state_dict'])
    self.rnn.eval()
    #self.feature_extraction_fn=self.checkpoint["feature_extraction_function"]
    self.feature_extraction_params=self.checkpoint["feature_extraction_parameters"]
    self.labels=self.checkpoint["labels"]

    # self.standard_labels=self.checkpoint['label_extraction_parameters']['ipa_ft_list']
    # self.ipa_ft_dict=self.checkpoint["label_extraction_parameters"]["ipa_ft_dict"]
    # self.ipa_list=self.checkpoint["label_extraction_parameters"]["ipa_symbol_list"]    
  def predict(self,item_fpath):
    times,ft_vector=self.feature_extraction_fn(item_fpath,self.feature_extraction_params)
    ft_tensor=torch.tensor(ft_vector,dtype=torch.float32)
    rnn_out= self.rnn(ft_tensor)
    preds0=out2labels(rnn_out.ravel(),self.labels)
    times_preds_list=[]
    for pr0,ti0 in zip(preds0,times):
      pr0=[(v[0],v[1].item()) for v in pr0]
      times_preds_list.append((ti0,pr0)) 
    return times_preds_list  


def predict_pos_frame(sent0,first_rnn0,second_rnn0,first_labels0,second_labels0,feature_extraction_parameters0):
  words0=sent0.split()
  raw_feature_list=[]
  for w0 in words0:
    cur_ft_list=extract_word_features(w0,feature_extraction_parameters0)
    raw_feature_list.append(cur_ft_list)
  input_tensor=torch.tensor(raw_feature_list,dtype=torch.float32)
  pos_rnn_out=first_rnn0(input_tensor)
  pos_predictions=out2labels(pos_rnn_out.ravel(),first_labels0)
  top_pos_predictions=[]
  for a in pos_predictions:
    top_pos_predictions.append(a[0][0])
  level2_feature_list=[]
  for i0, ft0 in enumerate(raw_feature_list):
    combined_features=ft0+list(pos_rnn_out[i0])
    level2_feature_list.append(combined_features)
  level2_input_tensor=torch.tensor(level2_feature_list,dtype=torch.float32)
  frame_rnn_out=second_rnn0(level2_input_tensor)
  frame_predictions=out2labels(frame_rnn_out.ravel(),second_labels0)
  final_output=[]
  for wd0,pos0, fp in zip(words0,top_pos_predictions,frame_predictions):
    if pos0=="Verb": 
      fp=[v for v in fp if v[0]!=""]
      top_frame=fp[0][0]
    else:
      top_frame=""
    final_output.append((wd0,pos0, top_frame))
    #print(wd0,pos0, top_frame)
  return final_output

