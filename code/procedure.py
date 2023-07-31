"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import torch
import numpy as np
import world
import utils
import multiprocessing
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import LightGCN
from collections import OrderedDict
import loss
import time

class Train():
    def __init__(self, loss_cal):
        self.loss = loss_cal
        self.test = Test()
        self.INFONCE = loss.InfoNCE_loss()
        self.BPR = loss.BPR()
        self.mse_loss = torch.nn.MSELoss()

    def train(self, sampler, Recmodel, epoch, optimizer, classifier):
        Recmodel:LightGCN = Recmodel
        batch_size = world.config['batch_size']
        dataloader = DataLoader(sampler, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)#每个batch为batch_size对(user, pos_item, neg_item), 见Dataset.__getitem__

        total_batch = len(dataloader)
        aver_loss = 0.
        aver_pop_acc = 0.

        for batch_i, train_data in tqdm(enumerate(dataloader), desc='training'):
            if world.config['sampling'] == 'uij':
                batch_users = train_data[0].long().to(world.device)
                batch_pos1 = train_data[1].long().to(world.device)
                batch_pos2 = None
                batch_neg = train_data[2].long().to(world.device)
            elif world.config['sampling'] == 'uii':
                batch_users = train_data[0].long().to(world.device)
                batch_pos1 = train_data[1].long().to(world.device)
                batch_pos2 = train_data[2].long().to(world.device)
            else:
                pass
            if world.config['projector'] == 'w':
                l_all, pop_acc_all = self.train_LightPopMix_w_project(Recmodel, batch_users, batch_pos1, batch_pos2, optimizer, epoch, classifier)
            elif world.config['projector'] == 'wo':
                l_all, pop_acc_all = self.train_LightPopMix_wO_project(Recmodel, batch_users, batch_pos1, batch_pos2, optimizer, epoch, classifier)

            aver_loss += l_all.cpu().item()
            aver_pop_acc += pop_acc_all.cpu().item()

        aver_loss = aver_loss / (total_batch)
        aver_pop_acc = aver_pop_acc / (total_batch)
        print(f'EPOCH[{epoch}]:loss {aver_loss:.3f}    pop_classifier_acc: {aver_pop_acc}')
        return aver_loss, aver_pop_acc
    
    def train_LightPopMix_w_project(self, Recmodel, batch_users, batch_pos1, batch_pos2, optimizer, epoch, classifier):
        all_users, all_items = Recmodel.computer()
        users_emb = all_users[batch_users]
        pos_emb1 = all_items[batch_pos1]
        pos_emb2 = all_items[batch_pos2]
        users_emb_ego = Recmodel.embedding_user(batch_users)
        pos_emb_ego1 = Recmodel.embedding_item(batch_pos1)
        pos_emb_ego2 = Recmodel.embedding_item(batch_pos2)

        # reg = (1/6)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/len(batch_users)
        reg = (0.5 * torch.norm(users_emb_ego) ** 2 + len(batch_users) * 0.5 * torch.norm(pos_emb_ego1) ** 2)/len(batch_users)

        ada_coef1 = self.loss.get_coef_adaptive(batch_users, batch_pos1, method='mlp', mode=world.config['centroid_mode'])
        ada_coef2 = self.loss.get_coef_adaptive(batch_users, batch_pos2, method='mlp', mode=world.config['centroid_mode'])

        pos_aug, ada_coef3 = self.mixup(pos_emb1, pos_emb2, ada_coef1, ada_coef2)
        # pos_aug, _ = self.mixup(pos_emb1, pos_emb2)

        loss_ada1 = self.loss.adaptive_loss(users_emb, pos_emb1, ada_coef1)
        loss_ada3 = self.loss.adaptive_loss(Recmodel.projector_user(users_emb), Recmodel.projector_item(pos_aug), ada_coef3)



        classifier_loss1, classifier_acc1 = classifier.cal_loss_and_test(pos_emb1.detach(), batch_pos1)
        classifier_loss2, classifier_acc2 = classifier.cal_loss_and_test(pos_emb2.detach(), batch_pos2)
        classifier_loss = (classifier_loss1 + classifier_loss2)*0.5
        classifier_acc = (classifier_acc1 + classifier_acc2)*0.5
        optimizer['pop'].zero_grad()
        classifier_loss.backward()
        optimizer['pop'].step()

        loss = world.config['weight_decay']*reg + loss_ada1 + loss_ada3*0.1
        optimizer['emb'].zero_grad()
        loss.backward()
        optimizer['emb'].step()        

        return loss, classifier_acc
        # all_users, all_items = Recmodel.computer()
        # users_emb = all_users[batch_users]
        # pos_emb1 = all_items[batch_pos1]
        # pos_emb2 = all_items[batch_pos2]
        # users_emb_ego = Recmodel.embedding_user(batch_users)
        # pos_emb_ego1 = Recmodel.embedding_item(batch_pos1)
        # pos_emb_ego2 = Recmodel.embedding_item(batch_pos2)

        # # reg = (1/6)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/len(batch_users)
        # reg = (0.5 * torch.norm(users_emb_ego) ** 2 + len(batch_users) * 0.5 * torch.norm(pos_emb_ego1) ** 2)/len(batch_users)

        # ada_coef1 = self.loss.get_coef_adaptive(batch_users, batch_pos1, method='mlp', mode=world.config['centroid_mode'])
        # ada_coef2 = self.loss.get_coef_adaptive(batch_users, batch_pos2, method='mlp', mode=world.config['centroid_mode'])

        # pos_aug, coef_aug, _ = self.mixup(pos_emb1, pos_emb2, ada_coef1, ada_coef2)

        # loss_ada1 = self.loss.adaptive_loss(users_emb, pos_emb1, ada_coef1)
        # loss_ada3 = self.loss.adaptive_loss(Recmodel.projector_user(users_emb), Recmodel.projector_item(pos_aug), coef_aug)

        # #pop classifier exp
        # classifier_loss1, classifier_acc1 = classifier.cal_loss_and_test(pos_emb1.detach(), batch_pos1)
        # classifier_loss2, classifier_acc2 = classifier.cal_loss_and_test(pos_emb2.detach(), batch_pos2)
        # classifier_loss = (classifier_loss1 + classifier_loss2)*0.5
        # classifier_acc = (classifier_acc1 + classifier_acc2)*0.5
        # optimizer['pop'].zero_grad()
        # classifier_loss.backward()
        # optimizer['pop'].step()

        # loss = world.config['weight_decay']*reg + loss_ada1 +  loss_ada3* world.config['lambda1']
        # optimizer['emb'].zero_grad()
        # loss.backward()
        # optimizer['emb'].step()
        

        return loss, classifier_acc

    def train_LightPopMix_wO_project(self, Recmodel, batch_users, batch_pos1, batch_pos2, optimizer, epoch, classifier):
        all_users, all_items = Recmodel.computer()
        users_emb = all_users[batch_users]
        pos_emb1 = all_items[batch_pos1]
        pos_emb2 = all_items[batch_pos2]
        users_emb_ego = Recmodel.embedding_user(batch_users)
        pos_emb_ego1 = Recmodel.embedding_item(batch_pos1)
        pos_emb_ego2 = Recmodel.embedding_item(batch_pos2)

        # reg = (1/6)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/len(batch_users)
        reg = (0.5 * torch.norm(users_emb_ego) ** 2 + len(batch_users) * 0.5 * torch.norm(pos_emb_ego1) ** 2)/len(batch_users)

        ada_coef1 = self.loss.get_coef_adaptive(batch_users, batch_pos1, method='mlp', mode=world.config['centroid_mode'])
        ada_coef2 = self.loss.get_coef_adaptive(batch_users, batch_pos2, method='mlp', mode=world.config['centroid_mode'])

        pos_aug, coef_aug, _ = self.mixup(pos_emb1, pos_emb2, ada_coef1, ada_coef2)

        loss_ada1 = self.loss.adaptive_loss(users_emb, pos_emb1, ada_coef1)
        loss_ada3 = self.loss.adaptive_loss(users_emb, pos_aug, coef_aug)

        #pop classifier exp
        classifier_loss1, classifier_acc1 = classifier.cal_loss_and_test(pos_emb1.detach(), batch_pos1)
        classifier_loss2, classifier_acc2 = classifier.cal_loss_and_test(pos_emb2.detach(), batch_pos2)
        classifier_loss = (classifier_loss1 + classifier_loss2)*0.5
        classifier_acc = (classifier_acc1 + classifier_acc2)*0.5
        optimizer['pop'].zero_grad()
        classifier_loss.backward()
        optimizer['pop'].step()

        loss = world.config['weight_decay']*reg + loss_ada1 +  loss_ada3* world.config['lambda1']
        optimizer['emb'].zero_grad()
        loss.backward()
        optimizer['emb'].step()
        

        return loss, classifier_acc

    
    def mixup(self, x1, x2, y1=None, y2=None):
        alpha = 2.
        beta = 2.
        size = [len(x1), 1]
        l = np.random.beta(alpha, beta, size)
        mixed_x = torch.tensor(l, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l, dtype=torch.float32).to(x2.device) * x2
        if y1 is None:
            return mixed_x, None, torch.tensor(l, dtype=torch.float32).to(x1.device)
        else:
            mixed_y = torch.tensor(l, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l, dtype=torch.float32).to(y2.device) * y2
            return mixed_x, mixed_y, torch.tensor(l, dtype=torch.float32).to(x1.device)
        
    def coef_mixup(self, epcoh):
        return 1/(epcoh+1)
    
    def advers_train(self, items_emb, batch_item, classifier, optimizer):
        #固定classifier的参数
        for param in classifier.parameters():
            param.requires_grad = False

        CE_loss, acc = classifier.cal_loss_and_test(items_emb, batch_item)

        #重新允许classifier参数更新
        for param in classifier.parameters():
            param.requires_grad = True

        return CE_loss


class Test():
    def __init__(self):
        pass
    
    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        #================Pop=================#
        groundTrue_popDict = X[2]#{0: [ [items of u1], [items of u2] ] }
        r, r_popDict = utils.getLabel(groundTrue, groundTrue_popDict, sorted_items)
        #================Pop=================#
        pre, recall, recall_pop, recall_pop_Contribute, ndcg = [], [], {}, {}, []
        num_group = world.config['pop_group']
        for group in range(num_group):
                recall_pop[group] = []
        for group in range(num_group):
                recall_pop_Contribute[group] = []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk(groundTrue, groundTrue_popDict, r, r_popDict, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])

            num_group = world.config['pop_group']
            for group in range(num_group):
                recall_pop[group].append(ret['recall_popDIct'][group])
            for group in range(num_group):
                recall_pop_Contribute[group].append(ret['recall_Contribute_popDict'][group])

            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))

        
        for group in range(num_group):
            recall_pop[group] = np.array(recall_pop[group])
        for group in range(num_group):
            recall_pop_Contribute[group] = np.array(recall_pop_Contribute[group])

        return {'recall':np.array(recall), 
                'recall_popDict':recall_pop,
                'recall_Contribute_popDict':recall_pop_Contribute,
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}


    def test(self, dataset, Recmodel, precal, epoch, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        testDict_pop = precal.popularity.testDict_PopGroup
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'recall_pop': {},
                'recall_pop_Contribute': {},
                'ndcg': np.zeros(len(world.config['topks']))}
        num_group = world.config['pop_group']
        for group in range(num_group):
            results['recall_pop'][group] = np.zeros(len(world.config['topks']))
            results['recall_pop_Contribute'][group] = np.zeros(len(world.config['topks']))

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            groundTrue_list_pop = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                #================Pop=================#
                groundTrue_pop = {}
                for group, ground in testDict_pop.items():
                    groundTrue_pop[group] = [ground[u] for u in batch_users]
                #================Pop=================#
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                #================Pop=================#
                groundTrue_list_pop.append(groundTrue_pop)
                #================Pop=================#
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list, groundTrue_list_pop)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                for group in range(num_group):
                    results['recall_pop'][group] += result['recall_popDict'][group]
                    results['recall_pop_Contribute'][group] += result['recall_Contribute_popDict'][group]
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            for group in range(num_group):
                results['recall_pop'][group] /= float(len(users))
                results['recall_pop_Contribute'][group] /= float(len(users))

            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if multicore == 1:
                pool.close()
            print(results)
            return results
    

    def valid_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall),
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}
    
    def valid_one_batch_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)

        k = world.config['topks'][0]
        ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
        recall = ret['recall']
        return recall

    def valid(self, dataset, Recmodel, multicore=0, if_print=True):
        u_batch_size = world.config['test_u_batch_size']
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'ndcg': np.zeros(len(world.config['topks']))}

        with torch.no_grad():
            users = list(validDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                if if_print:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [validDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [ 
                #         utils.AUC(rating[i],
                #                   dataset, 
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.valid_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.valid_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if multicore == 1:
                pool.close()
            if if_print:
                print('VALID',results)
            return results


    def valid_batch(self, dataset, Recmodel, batch_users):
        batch_users = batch_users.cpu()
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])

        with torch.no_grad():
            users = list(batch_users)
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [validDict[u.item()] for u in batch_users]
            batch_users_gpu = batch_users.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            recall = self.valid_one_batch_batch([rating_K.cpu(), groundTrue])
            return recall/float(len(users))