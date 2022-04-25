import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import GPT2Tokenizer, GPT2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TOKENS = 50257
GOAL_LENGTH = 32
HISTORY_LENGTH = 128
NAMES_LENGTH = 256*2

class VHDataloader:
    class VHStructuredDataset(Dataset):
        def __init__(self, data_pth mini_batch = False):
            self.data = {'goal': None,
                        'history': None,
                        'obs_ids': None,
                        'obs_names': None,
                        'obs_states': None,
                        'obs_positions': None,
                        'char_label': None,
                        'action_label': None,
                        'object_label': None}
            self.mini_batch = mini_batch

            for key in self.data:
                self.data[key] = np.load(os.path.join(data_pth,"{}.npy".format(key)), allow_pickle=True)
                
        def __getitem__(self, index):
            goal = self.data['goal'][index]
            history = self.data['history'][index]
            obs_ids = self.data['obs_ids'][index]
            obs_names = self.data['obs_names'][index]
            obs_states = self.data['obs_states'][index]
            obs_positions = self.data['obs_positions'][index]
            char_label = self.data['char_label'][index]
            action_label = self.data['action_label'][index]
            object_label = self.data['object_label'][index]

            return {'goal': goal,
                    'history': history,
                    'obs_ids': obs_ids,
                    'obs_names': obs_names,
                    'obs_states': obs_states,
                    'obs_positions': obs_positions,
                    'char_label':char_label,
                    'action_label':action_label,
                    'object_label':object_label}
            
        def __len__(self):
            return len(self.data['goal']) if not self.mini_batch else 10


    def __init__(self, data_pth, batch_size = 16, num_workers = 2, mini_batch = False):
        with open(os.path.join(data_pth, 'relationships'), 'r') as f:
            self.relationships =  json.load(f)
        self.structured_dataset = self.VHStructuredDataset(data_pth, mini_batch)
        self.dataloader = DataLoader(self.structured_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)
        self.batch_size = batch_size
    
    def __iter__(self):
        for batch_n, sample in enumerate(self.dataloader):
            sample['relationships'] =  \
               self.relationships[batch_n*self.batch_size:(batch_n+1)*self.batch_size]
            yield sample
    
    def __len__(self):
        return len(self.structured_dataset)

def get_tokenizer(lm):
    tokenizer = GPT2Tokenizer.from_pretrained(lm)
    return tokenizer

def get_word_embeddings(lm):
    model = GPT2Model.from_pretrained(lm)
    word_embeddings = model.wte.weight
    return word_embeddings

def embed(sentence, tokenizer, word_embeddings, max_len=128):
    with torch.no_grad():
        #list sentence arg can be like ['I have a pen', 'I have an apple']
        tokenized = tokenizer(sentence, max_length=max_len, pad_to_max_length=True, return_tensors="pt").input_ids.to(device)
        one_hot_tokenized = F.one_hot(tokenized, num_classes=NUM_TOKENS).type(torch.float32)
        embeddings = one_hot_tokenized @ word_embeddings
    return embeddings

def preprocess(data, tokenizer, embeddings):
    data['goal'] = embed(data['goal'], tokenizer, embeddings, max_len = GOAL_LENGTH)
    data['history'] = embed(data['history'], tokenizer, embeddings, max_len = HISTORY_LENGTH)
    data['obs_names'] = embed(data['obs_names'], tokenizer, embeddings, max_len = NAMES_LENGTH)
    data['obs_states'] = data['obs_states'].type(torch.float32).to(device)
    data['obs_positions'] = data['obs_positions'].type(torch.float32).to(device)
    data['char_label'] = data['char_label'].to(torch.int64).squeeze().to(device)
    data['action_label'] = data['action_label'].to(torch.int64).squeeze().to(device)
    data['object_label'] = data['object_label'].to(torch.int64).squeeze().to(device)
    return data


class LMDecisionMaker(nn.Module):
    def __init__(self):
        super().__init__()
        # define the layer in LM framework
        self.state_fc = nn.Linear(6, 3*768).to(device)
        self.position_fc1 = nn.Linear(6, 64).to(device)
        self.relu = nn.ReLU(inplace = True)
        self.position_fc2 = nn.Linear(64, 3*768).to(device)
        self.observation_fc = nn.Linear(7*768, 768).to(device)
        self.action_fc = nn.Linear(768, 8).to(device)
        self.char_fc = nn.Linear(768, 2).to(device)
        self.object1_fc = nn.Linear(512, 512).to(device)
        self.object2_fc = nn.Linear(512, 512).to(device)
        self.lm = GPT2Model.from_pretrained(pretrained_lm).to(device)
        self.alpha = 5
    
    def lm_infer(self, input):
        return self.lm(inputs_embeds=input).last_hidden_state

    def forward(self,
                goal,
                history,
                obs_ids,
                obs_names,
                obs_states,
                obs_positions,
                char_label,
                action_label,
                object_label,
                relationships = [],
                mode='train'):
        
        goal_embeddings = goal
        
        history_embeddings = history
        
        #name_embeddings = self.name_fc(obs_names).unsqueeze(2) #N*L*768
        name_embeddings = obs_names

        state_embeddings = self.state_fc(obs_states) #N*L*768
        
        position_embeddings = self.position_fc2(self.relu(self.position_fc1(obs_positions))) #N*L*768
    
        observation_embeddings = torch.cat((name_embeddings, state_embeddings, position_embeddings), 2)
        observation_embeddings = observation_embeddings.view(-1, NAMES_LENGTH, 7*768)
        observation_embeddings = self.observation_fc(observation_embeddings)
        observation_feature = self.lm_infer(observation_embeddings) # N*L*768
        
        contextualized_embeddings = torch.cat((goal_embeddings, history_embeddings, observation_embeddings),1)
        contextualized_embeddings = self.lm_infer(contextualized_embeddings)
        contextualized_feature = torch.mean(contextualized_embeddings, 1) # N*768
        
        char_scores = self.char_fc(contextualized_feature) # N * 2
        char_prediction = torch.argsort(char_scores, dim=1, descending=True) # N*10

        action_scores = self.action_fc(contextualized_feature) # N*10
        action_prediction = torch.argsort(action_scores, dim=1, descending=True) # N*10
        
        repeated_cf = contextualized_feature.unsqueeze(1).repeat(1,observation_feature.shape[1], 1)
        object_features = torch.sum(observation_feature*repeated_cf, axis = -1)
        object1_scores = self.object1_fc(object_features)
        object2_scores = self.object2_fc(object_features)
        object1_prediction = torch.argsort(object1_scores, dim=1, descending=True)
        object2_prediction = torch.argsort(object2_scores, dim=1, descending=True)
        #object_scores = torch.sum(observation_feature*repeated_cf, axis = -1) # N*L
        #object_prediction = torch.argsort(object_scores, dim=1, descending=True)

        if mode == 'train':
            loss = self.compute_loss(char_scores,
                                     action_scores,
                                     object1_scores,
                                     object2_scores,
                                     char_label,
                                     action_label,
                                     object_label)
 
            accuracy = self.compute_accuracy(char_prediction[:,0],
                                             action_prediction[:,0],
                                             object1_prediction[:,0],
                                             object2_prediction[:,0],
                                             char_label,
                                             action_label,
                                             object_label)
            # simple accuracy calculation, different from inference time
            return loss, accuracy

        next_predicate = self.predictor(char_prediction,
                                        action_prediction,
                                        object1_prediction,
                                        object2_prediction,
                                        obs_ids,
                                        obs_names,
                                        relationships)

        return next_predicate

    def predictor(self,
                  char_prediction,
                  action_prediction,
                  object1_prediction,
                  object2_prediction,
                  obs_ids,
                  obs_names,
                  relationships = []):
        """
        Compute the loss and accuracy

        action_prediction: a torch tensor with a size (10), showing the probability of 10 actions
        object_prediction: a torch tensor with a size (len(observations)), showing the probability of all visible objects
        observations: a dictionary mapping the visible object (including all rooms) id to name, state vector and position vector
        relationships: a dictionary containing 4 relationship dictionary: sitting, close, inside, hold

        return: a predicate string in the form of VirtualHome format without character, e.g. '[walk] <chair> (1)'
        """
        pass
        return None

    def compute_loss(self,
                     char_scores,
                     action_scores,
                     object1_scores,
                     object2_scores,
                     char_label,
                     action_label,
                     object_label):
        """
        Compute the cross entropy loss
        scores: consists of action, char and label scores
        labels: consists of action, char and label label

        return: the cross entropy loss of the action and objects
        action_index = {'standup': 0, 'walk': 1, 'sit': 2, 'grab': 3, 'open': 4, 'close': 5, 'switchon': 6, 'switchoff': 7, 'put': 8, 'putin': 9}

        """

        char_loss = F.cross_entropy(char_scores, char_label)
        action_loss = F.cross_entropy(action_scores, action_label)
        count_ind = (action_label > 5).to(torch.float32) # only apply to obj2
        object_loss = F.cross_entropy(object1_scores, object_label[:,0], reduction = 'none') \
                    + count_ind * F.cross_entropy(object2_scores, object_label[:,1], reduction = 'none')
        object_loss = object_loss.mean()
        
        loss = char_loss + action_loss + object_loss
        return loss

    def compute_accuracy(self,
                         char_pred,
                         action_pred,
                         object1_pred,
                         object2_pred,
                         char_label,
                         action_label,
                         object_label):
        """
        Compute the accuracy

        next_index: a tuple with three torch tensors with size (1), each indicates the index of the action and objects
        action label: a torch tensor of size (1), showing the true action label
        object label: a torch tensor of size (0) or (1) or (2), showing the true object labels

        return: the cross entropy loss of the action and objects
        """
        char_correct = torch.eq(char_pred, char_label).float()
        action_correct = torch.eq(action_pred, action_label).float()

        ignore_ind = (action_label <= 5).to(torch.float32)# only apply to obj2
        object1_correct = torch.eq(object1_pred, object_label[:,0]).float()
        object2_correct = torch.eq(object2_pred, object_label[:,1]).float()
        object2_correct = torch.logical_xor(ignore_ind, object2_correct).float()

        return 0.25*(char_correct.mean() \
             + action_correct.mean() \
             + object1_correct.mean() \
             + object2_correct.mean() )

