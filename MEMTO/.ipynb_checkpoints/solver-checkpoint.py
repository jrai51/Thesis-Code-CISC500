# Some code based on https://github.com/thuml/Anomaly-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.Transformer import TransformerVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #only one gpu on my machine
device_ids = [0]

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        


class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss
        
        # Check memory usage before emptying the cache
        reserved_before = torch.cuda.memory_reserved()
        allocated_before = torch.cuda.memory_allocated()

        torch.cuda.empty_cache() # BECAUSE GPU KEEPS RUNNING OUT OF MEMORY. MAY HAVE TO REMOVE THIS.
        # Check memory usage after emptying the cache
        reserved_after = torch.cuda.memory_reserved()
        allocated_after = torch.cuda.memory_allocated()
        print("gpu cache emptied.")
        print(f"Reserved before: {reserved_before / 1e9:.2f} GB, after: {reserved_after / 1e9:.2f} GB")
        print(f"Allocated before: {allocated_before / 1e9:.2f} GB, after: {allocated_after / 1e9:.2f} GB")



class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)

        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = self.vali_loader
        
        if self.memory_initial == "False":
            
            self.memory_initial = False
        else:
            self.memory_initial = True


        self.memory_init_embedding = None


        self.build_model(memory_init_embedding=self.memory_init_embedding)
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        # file_handler = logging.FileHandler(f'./hyperparameters_tuning/memory_item_numbers/number_{self.dataset}.log')
        # file_handler.setFormatter(formatter)
        # self.logger.addHandler(file_handler)

    def build_model(self,memory_init_embedding):
        
        self.model = TransformerVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, \
                                    e_layers=3, d_model=self.d_model, n_memory=self.n_memory, device=self.device, \
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding, phase_type=self.phase_type, dataset_name=self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids, output_device=0).to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = [] ; valid_re_loss_list = [] ; valid_entropy_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items, attn = output_dict['out'], output_dict['queries'], output_dict['mem'], output_dict['attn']
            
            rec_loss = self.criterion(output, input)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd*entropy_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)
    
    def _get_min_max(self, data_loader):
        """ Gets the minimum and maximum values for each feature in the data loader """
        

    def train(self, training_type):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=10, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)

        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []; entropy_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)    
                output_dict = self.model(input_data)
                
                output, memory_item_embedding, queries, mem_items, attn = output_dict['out'], output_dict['memory_item_embedding'], output_dict['queries'], output_dict["mem"], output_dict['attn']

                rec_loss = self.criterion(output, input)
                entropy_loss = self.entropy_loss(attn)
                loss = rec_loss + self.lambd*entropy_loss
                
                loss_list.append(loss.detach().cpu().numpy())
                entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                try:
                    loss.mean().backward()
                    
                except:
                    import pdb; pdb.set_trace()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)
            valid_loss , valid_re_loss_list, valid_entropy_loss_list = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss))

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return memory_item_embedding
    
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
        self.model.eval()
        
        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        train_attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input_data)
            
#             print("Output Dict:", output_dict)

            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input,output),dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            train_attens_energy.append(cri)

        train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
        train_energy = np.array(train_attens_energy)

        valid_attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input,output),dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            print("Reconstruction loss:", rec_loss)
            print("latent score:", latent_score)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            valid_attens_energy.append(cri)

        valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)
        valid_energy = np.array(valid_attens_energy)

        combined_energy = np.concatenate([train_energy, valid_energy], axis=0)
        print(f"train val combined_energy: {combined_energy}")

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)


        distance_with_q = []
        reconstructed_output = []
        original_output = []
        rec_loss_list = []

        test_labels = []
        test_attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output_dict= self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input,output),dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            loss = latent_score * rec_loss

            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels)

            d_q = gathering_loss(queries, mem_items)*rec_loss
            distance_with_q.append(d_q.detach().cpu().numpy())
            distance_with_q.append(gathering_loss(queries, mem_items).detach().cpu().numpy())

            reconstructed_output.append(output.detach().cpu().numpy())
            original_output.append(input.detach().cpu().numpy())
            rec_loss_list.append(rec_loss.detach().cpu().numpy())

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)

        reconstructed_output = np.concatenate(reconstructed_output,axis=0).reshape(-1)
        original_output = np.concatenate(original_output,axis=0).reshape(-1)
        rec_loss_list = np.concatenate(rec_loss_list,axis=0).reshape(-1)


        #reconstruct_path = f"./hyperparameters_tuning/reconstruction/{self.dataset}_"
        #np.save(reconstruct_path+'reconstructed_output', reconstructed_output)
        #np.save(reconstruct_path+'original_output', original_output)
        #np.save(reconstruct_path+'rec_loss',rec_loss_list)
        #np.save(reconstruct_path+'gt_labels',test_labels)
        #np.save(reconstruct_path+'anomaly_score_only_gathering_loss',test_energy)
        
        distance_with_q = np.concatenate(distance_with_q,axis=0).reshape(-1)

        normal_dist = []
        abnormal_dist = []
        for i,l in enumerate(test_labels):
            if l == 0:
                normal_dist.append(distance_with_q[i])
            else:
                abnormal_dist.append(distance_with_q[i])

        #dist_path = f"./hyperparameters_tuning/norm_abnorm_distribtuion/{self.dataset}_"
        #normal_dist = np.array(normal_dist)
        #abnormal_dist = np.array(abnormal_dist)

        #np.save(dist_path+'normal_dist_only_gl', normal_dist)
        #np.save(dist_path+'abnormal_dist_only_gl', abnormal_dist)
        
        print(f"test_energy: {test_energy}")

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

#         anomaly_state = False
#         for i in range(len(gt)):
#             if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#                 anomaly_state = True
#                 for j in range(i, 0, -1):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#                 for j in range(i, len(gt)):
#                     if gt[j] == 0:
#                         break
#                     else: 
#                         if pred[j] == 0:
#                             pred[j] = 1
#             elif gt[i] == 0:
#                 anomaly_state = False
#             if anomaly_state:
#                 pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print(pred)
        print("gt:   ", gt.shape)
        print(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        print('='*50)

        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(precision,4)}")
        self.logger.info(f"Recall: {round(recall,4)}")
        self.logger.info(f"f1_score: {round(f_score,4)} \n")
        return accuracy, precision, recall, f_score

    def get_memory_initial_embedding(self,training_type='second_train'):

        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth')))
        self.model.eval()
        
        for i, (input_data, labels) in enumerate(self.k_loader):

            input = input_data.float().to(self.device)
            if i==0:
                output= self.model(input)['queries']
            else:
                output = torch.cat([output,self.model(input)['queries']], dim=0)
        
        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory, d_model=self.d_model)

        self.memory_initial = False

        self.build_model(memory_init_embedding = self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[:int(self.n_memory),:]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')

        torch.save(memory_item_embedding, item_path)
        
        
    def calculate_windowed_threshold(self, window_size):
        self.model.eval()
        thresholds = []
        criterion = nn.MSELoss(reduce=False)
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        # Combine train and validation data
        combined_loader = torch.utils.data.ConcatDataset([self.train_loader.dataset, self.vali_loader.dataset])
        combined_loader = torch.utils.data.DataLoader(combined_loader, batch_size=window_size, shuffle=False)

        with torch.no_grad():
            for window_data, _ in combined_loader:
                print(f"train window data shape: {window_data.shape}")
                window_data = window_data.float().to(self.device)

                # Reconstruct the window through the model
                output_dict = self.model(window_data)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

                # Calculate reconstruction loss and latent score
                rec_loss = torch.mean(criterion(window_data, output), dim=-1)
                latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
                loss = latent_score * rec_loss

                # Calculate threshold for this window
                window_energy = loss.cpu().numpy()
                window_threshold = np.percentile(window_energy, 100 - self.anormly_ratio)
                thresholds.append(window_threshold)

        # Return the mean of the thresholds
        return np.mean(thresholds)



        
    def test_with_windows(self, csv_path="data/WACA/WACA/test.csv", thresh=3e+20, window_size=1000, anomaly_flag=True):
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Extract features (x, y, z)
        features = df[['timestamp', 'x', 'y', 'z']].values

        # Segment data into windows
        num_windows = len(features) // window_size
        windows = np.array_split(features[:num_windows * window_size], num_windows)

        # Load the trained model
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
        self.model.eval()
#         thresh = self.calculate_windowed_threshold(window_size)

        # Need to calculate threshold from training data windows.
        print(f"threshold = {thresh}")

        window_accuracies = []

        for i, window in enumerate(windows):
            # Prepare input data
            input_data = torch.FloatTensor(window).unsqueeze(0).to(self.device)

            # Get model output
            with torch.no_grad():
                output_dict = self.model(input_data)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            # Calculate reconstruction loss
            rec_loss = torch.mean(nn.MSELoss(reduction="none")(input_data, output), dim=-1)

            # Calculate anomaly scores
            gathering_loss = GatheringLoss(reduce=False)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
            anomaly_scores = (latent_score * rec_loss).squeeze().detach().cpu().numpy()
            
#             print(anomaly_scores)
            
            # Determine threshold (you may want to adjust this based on your specific needs)
#             thresh = 0.0125 # This has been hardcoded based on prior tests

            # Classify points as anomalous or normal
            predictions = (anomaly_scores > thresh).astype(int)
            print(predictions)

            # Calculate accuracy
#             print(f"anomaly_flag, {anomaly_flag}")
            true_labels = np.full(len(predictions), int(anomaly_flag))
            accuracy = accuracy_score(true_labels, predictions)
#             print(f"accuracy: {accuracy}")

            window_accuracies.append({
                'window_index': i,
                'accuracy': accuracy,
                'num_anomalies': sum(predictions),
                'total_points': len(predictions)
            })
            
        return window_accuracies
    
    def inference_with_window(self, anomaly_threshold=0.5):
        """
        Inference method for MEMTO that processes test data in non-overlapping windows.

        The method performs two main steps:
          1. It computes a fixed anomaly threshold using windows from the training set.
             For each training window, it computes the reconstruction loss and latent score 
             (via GatheringLoss), and then derives an "energy" for each data point. All energies 
             are concatenated and the threshold is set as the (100 - anormly_ratio) percentile.
          2. It then processes the test data batch-by-batch, splits each batch into non-overlapping 
             windows, computes the energy for each window in the same way, and flags the window as 
             anomalous if the fraction of points with energy above the fixed threshold exceeds 
             (anomaly_threshold/100).

        Args:
            anomaly_threshold (float): Minimum percentage (e.g., 2 for 2%) of anomalous points in a 
                                       window required to flag that window as anomalous.

        Returns:
            window_results (list): List of dictionaries with window-level metrics and decisions.
        """
        # Load the MEMTO checkpoint (from second training)
        checkpoint_path = os.path.join(str(self.model_save_path), f"{self.dataset}_checkpoint_second_train.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        # Set up loss functions and temperature.
        criterion = nn.MSELoss(reduction="none")
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        # ---------------------- Step 1: Compute Fixed Threshold ----------------------
        train_energy_list = []
        # Process each batch in the training loader.
        for i, (input_data, _) in enumerate(self.train_loader):
            batch_size, seq_len, num_features = input_data.shape
            num_windows = seq_len // self.win_size
            if num_windows < 1:
                continue
            # Split batch into non-overlapping windows.
            input_windows = input_data[:, :num_windows * self.win_size, :].reshape(batch_size * num_windows, self.win_size, num_features)
            for w in range(input_windows.shape[0]):
                window_input = input_windows[w:w+1].float().to(self.device)
                output_dict = self.model(window_input)
                # MEMTO returns keys: 'out', 'queries', 'mem'
                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']
                rec_loss = torch.mean(criterion(window_input, output), dim=-1)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
                loss = latent_score * rec_loss
                energy = loss.detach().cpu().numpy().flatten()
                train_energy_list.append(energy)
        if len(train_energy_list) == 0:
            print("No training windows processed!")
            return []
        train_energy = np.concatenate(train_energy_list, axis=0)
        fixed_thresh = np.percentile(train_energy, 100 - self.anormly_ratio)
        print("Fixed threshold from training windows:", fixed_thresh)

        # ---------------------- Step 2: Process Test Windows ----------------------
        window_results = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            batch_size, seq_len, num_features = input_data.shape
            num_windows = seq_len // self.win_size
            if num_windows < 1:
                continue
            # Split the batch into non-overlapping windows.
            input_windows = input_data[:, :num_windows * self.win_size, :].reshape(batch_size * num_windows, self.win_size, num_features)
            label_windows = labels[:, :num_windows * self.win_size].reshape(batch_size * num_windows, self.win_size)
            for w in range(input_windows.shape[0]):
                window_input = input_windows[w:w+1].float().to(self.device)
                window_labels = label_windows[w]
                output_dict = self.model(window_input)
                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']
                rec_loss = torch.mean(criterion(window_input, output), dim=-1)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
                loss = latent_score * rec_loss
                energy = loss.detach().cpu().numpy().flatten()
                # Classify each point as anomalous if its energy exceeds the fixed threshold.
                pred = (energy > fixed_thresh).astype(int)
                anomaly_percentage = np.sum(pred) / len(pred)
                # Flag the window as anomalous if the anomaly percentage meets the given threshold.
                decision = anomaly_percentage >= (anomaly_threshold / 100)
                gt = window_labels.numpy().flatten().astype(int)
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                acc = accuracy_score(gt, pred)
                prec, rec, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
                window_results.append({
                    'window_idx': w,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f_score': f_score,
                    'anomaly_percentage': anomaly_percentage,
                    'anomaly_detected': decision
                })
                print(f"Window {w}: accuracy={acc:.4f}, anomaly_percentage={anomaly_percentage:.5f}, imposter decision={decision}")

        return window_results
    
    
    def inference_with_window_sums(self, anomaly_threshold=0.5):
        """
        Inference method for MEMTO that processes test data in non-overlapping windows.

        The method processes the test data batch-by-batch, splits each batch into non-overlapping 
             windows, computes the energy for each window, then returns the sum and average energies

        Args:
            anomaly_threshold (float): Minimum energy score for a window to be flagged as an anomaly
        Returns:
            window_results (list): List of dictionaries with window-level metrics and decisions.
        """
        # Load the MEMTO checkpoint (from second training)
        checkpoint_path = os.path.join(str(self.model_save_path), f"{self.dataset}_checkpoint_second_train.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        # Set up loss functions and temperature.
        criterion = nn.MSELoss(reduction="none")
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

       
        # ---------------------- Step 1: Process Test Windows ----------------------
        window_results = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            batch_size, seq_len, num_features = input_data.shape
            num_windows = seq_len // self.win_size
            if num_windows < 1:
                continue
            # Split the batch into non-overlapping windows.
            input_windows = input_data[:, :num_windows * self.win_size, :].reshape(batch_size * num_windows, self.win_size, num_features)
            label_windows = labels[:, :num_windows * self.win_size].reshape(batch_size * num_windows, self.win_size)
            for w in range(input_windows.shape[0]):
                window_input = input_windows[w:w+1].float().to(self.device)
                window_labels = label_windows[w]
                output_dict = self.model(window_input)
                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']
                rec_loss = torch.mean(criterion(window_input, output), dim=-1)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
                loss = latent_score * rec_loss
                energy = loss.detach().cpu().numpy().flatten()
                
                # Calculate energy sum and energy mean.
                energy_sum = np.sum(energy)
                energy_mean = np.mean(energy) 
                
                # Flag the window as anomalous if the anomaly percentage meets the given threshold.
                
                window_results.append({
                    'window_idx': w,
                    'energy_sum': energy_sum,
                })
                print(f"Window {w}: energy_sum={energy_sum:.5f}")

        return window_results

    
   
