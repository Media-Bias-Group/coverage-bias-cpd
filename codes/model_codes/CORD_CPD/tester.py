import math
import sys, os
import os.path as osp
import time
import pickle
import psutil

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.exp_utils import create_exp_dir
from utils.utils import *
from utils.cp_data_utils import *
from utils.cp_function_utils import *
from models import cord_cpd

#dataset = ["loc", "vel", "edge"]
dataset = ["gdelt"]

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args

        self.args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.args.cuda else torch.device("cpu")
        self.args.factor = not args.no_factor
        self.exp_dir = args.exp_dir
        self.logging = create_exp_dir("exp/test_res")

        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def load_data(self, data_path):
        args = self.args
        if args.data_type == "real":
            self.train_loader, self.valid_loader, self.test_loader = load_cp_real_data(data_path,args.batch_size,data_norm=args.data_norm)
        else:
            self.train_loader, self.valid_loader, self.test_loader = load_cp_data(data_path,args.batch_size,args.suffix,data_norm=args.data_norm)
        #self.train_loader, self.valid_loader, self.test_loader = load_cp_data(
        #    data_path, args.batch_size, args.suffix, data_norm=args.data_norm)
        off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)

    def set_model(self):
        self.model = cord_cpd.Model(self.args)
        self.model.to_device(self.device)
        self.model.load(self.exp_dir)

    def solve(self):
        base_path = self.args.data_path
        self.set_model()
        self.logging("\n *******************************************")
        self.logging(self.exp_dir)
        print(f"dir: {self.exp_dir}, base path: {self.args.data_path}")
        for d in dataset:
            #data_path = osp.join(base_path, f"cp_{d}")
            data_path = base_path #test on the same as training dataset
            print(f"data path: {data_path}")
            self.load_data(data_path)
            self.test('valid', d)
            self.test('test', d) #uncommented
            self.test('train', d) #uncommented

    @torch.no_grad()
    def test(self, mode='test', datatype='edge'):
        process = psutil.Process(os.getpid())
        start_time = time.time()
        print("starting test")
        acc_test = []
        mse_test = []
        probs = []
        cpds = []
        relations = []
        origs, recons = [], []

        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'test':
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader
        self.model.set_eval()
        for batch_idx, (data, relation, cpd) in enumerate(data_loader):

            current_rss = process.memory_info().rss
            if batch_idx == 0:
                peak_rss = current_rss
            peak_rss = max(peak_rss, current_rss)

            data, relation = data.to(self.device), relation.to(self.device)
            # data [batch_size, num_atoms, num_timesteps, num_dims]
            # relations [batch_size, time_step, edge]
            # data = data[:, :, :self.args.timesteps, :]

            logits = self.model.encode(data, self.rel_rec, self.rel_send)

            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)

            edges = F.gumbel_softmax(logits, tau=self.args.temp, hard=True)

            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)
            
            prob = F.softmax(logits, -1)
            probs.append(prob.detach().cpu().numpy())
            cpds.append(cpd.numpy())
            relations.append(relation.detach().cpu().numpy())
            origs.append(data.transpose(1, 2).contiguous().detach().cpu().numpy())

            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)

            # validation output uses teacher forcing
            output = self.model.decoder(data, edges, self.rel_rec, self.rel_send, 1)

            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)

            recon = self.model.decoder.forward_reconstruct(data, edges, self.rel_rec, self.rel_send)
            recons.append(recon.detach().cpu().numpy())
            target = data[:, :, 1:, :]

            target = target[:, :, self.args.begin_steps:, :]
            output = output[:, :, self.args.begin_steps:, :]

            loss_mse = F.mse_loss(output, target) / (2 * self.args.var) * 400

            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)

            #acc = edge_accuracy(logits, relation, begin_steps=25, end_steps=-25)
            if self.args.data_type == "sim":
                acc = edge_accuracy(logits, relation, begin_steps=25, end_steps=-25) # changed to not calculate accuracy for real data
            else:
                acc = np.nan

            acc_test.append(acc)
            mse_test.append(loss_mse.item())


        probs = np.concatenate(probs)
        cpds = np.concatenate(cpds)
        relations = np.concatenate(relations)
        recons = np.concatenate(recons)
        origs = np.concatenate(origs)

        save_dir = osp.join(self.exp_dir, "{}_{}_dict.npy".format(datatype, mode))
        #dic = {"probs": probs, "cpds": cpds, "relations": relations, "origs": origs, "recons": recons}
        #np.save(save_dir, dic)

        self.logging('--------------------------------')
        self.logging('--------{}--{}-------------'.format(mode, datatype))
        self.logging('--------------------------------')
        self.logging('mse: {:.10f}, acc: {:.10f}'.format(np.mean(mse_test), np.mean(acc_test)))

        type2_score = cal_cp_from_output(probs)
        #avg_roc, avg_dist, avg_tri = cpd_metrics(type2_score, cpds, anomaly_input=True)
        if self.args.data_type == "sim":
            avg_roc, avg_dist, avg_tri = cpd_metrics(type2_score, cpds, anomaly_input=True)
        else:
            avg_roc, avg_dist, avg_tri = np.nan, np.nan, np.nan

        pred_cpd = cal_cp_from_output(probs, cal_cp)
        pred_cpd_continuous = cal_cp_from_output(probs, cal_cp_continuous)
        #print("pred_cpd: ", pred_cpd, pred_cpd_continuous)
        #roc_score = score(pred_cpd, cpds)
        #roc_score_continuous = score(pred_cpd_continuous, cpds)
        #print("roc score: ", roc_score)

        self.logging("roc continuous: {}, dist: {}, tri: {}\n".format(avg_roc, avg_dist, avg_tri))
        mse_scores = [mse_anomaly(recons, origs, step=i) for i in range(1, 6)]

        if self.args.data_type == "sim":
            mse_res = [cpd_metrics(mse_scores[i], cpds, anomaly_input=True) for i in range(5)]
            for i, r in enumerate(mse_res):
                self.logging("future step {}, reconstruction quality {}".format(i+1, r))

        type1_score = mse_scores[-1]
        combined = anomaly_combined_score(type1_score, type2_score)

        #self.logging("combined score: {}".format(cpd_metrics(combined, cpds, anomaly_input=True)))
        if self.args.data_type == "sim":
            self.logging("combined score: {}".format(cpd_metrics(combined, cpds, anomaly_input=True)))

        c1, c2 = change_type_classification(type1_score, type2_score)
        if "edge" in datatype:
            self.logging("accuracy: {}".format(c2 / (c1 + c2)))
        else:
            self.logging("accuracy: {}".format(c1 / (c1 + c2)))
        # print('MSE: {}'.format(mse_str))

        self.logging("relation: & {:.4f} & {:.2f} & {:.4f}".format(avg_roc, avg_dist, avg_tri))
        if self.args.data_type == "sim":
            mse_roc, mse_dist, mse_tri = cpd_metrics(type1_score, cpds, anomaly_input=True)
            self.logging("mse: & {:.4f} & {:.2f} & {:.4f}".format(mse_roc, mse_dist, mse_tri))
            com_roc, com_dist, com_tri = cpd_metrics(combined, cpds, anomaly_input=True)
            self.logging("combine: & {:.4f} & {:.2f} & {:.4f}".format(com_roc, com_dist, com_tri))

        pred_relation_cp = np.argmax(type2_score, axis=-1)
        pred_mse_cp = np.argmax(type1_score, axis=-1)
        pred_combined_cp = np.argmax(combined, axis=-1)

        dic = {
            "probs": probs,
            "cpds": cpds,
            "relations": relations,
            "origs": origs,
            "recons": recons,
            "type2_score": type2_score,
            "type1_score": type1_score,
            "combined_score": combined,
            "pred_relation_cp": pred_relation_cp,
            "pred_mse_cp": pred_mse_cp,
            "pred_combined_cp": pred_combined_cp,
        }
        np.save(save_dir, dic)

        end_time = time.time()

        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            print(f"Peak     : {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        else:
            print("CUDA is not available.")

        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024**2:.1f} MB")  # Resident Set Size
        print(f"Peak memory usage: {peak_rss / 1024**2:.1f} MB")  # Peak RSS

        print(f"Fished testing in {end_time-start_time:2f} seconds")



