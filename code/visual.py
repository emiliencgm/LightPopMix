
import os
import torch
import numpy as np
import world
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataloader import dataset
from model import LightGCN
from precalcul import precalculate
from utils import randint_choice
import seaborn as sns
import matplotlib.colors as mcolors
'''
green: #D1ECB9
blue: #6DC3F2
purple: #A68BC2
orange: #FFB265

zhihu green: #8ECFC9
'''

class Quantify():
    def __init__(self, dataset:dataset, model:LightGCN, precal:precalculate) -> None:
        self.dataset = dataset
        self.model = model
        self.precal = precal

    def visualize_tsne(self, epoch, figsize=(25,25)):
        '''
        embedding: Recmodel.item_embedding
        label: np.array([pop1, pop2])
        '''
        world.cprint('[t-SNE]')
        with torch.no_grad():
            #只展示最热门和最冷门
            # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
            # plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号
            items = self.model.embedding_item.weight.cpu()
            users = self.model.embedding_user.weight.cpu()
            keep_idx_item = torch.zeros(len(items))
            keep_idx_user = torch.zeros(len(users))
            label_item = self.precal.popularity.item_pop_group_label.copy()
            label_user = self.precal.popularity.user_pop_group_label.copy()

            for item in range(len(items)):
                if label_item[item] in world.config['tsne_group']:
                    keep_idx_item[item] = 1.
                else:
                    pass
            for user in range(len(users)):
                if label_user[user] in world.config['tsne_group']:
                    keep_idx_user[user] = 1.
                else:
                    pass

            keep_idx_item = keep_idx_item.to(torch.bool)
            keep_idx_user = keep_idx_user.to(torch.bool)

            items = items[keep_idx_item]
            users = users[keep_idx_user]
            label_item = label_item[keep_idx_item]
            label_user = label_user[keep_idx_user]

            #随机采样其中的2000个点
            #keep_random1 = np.random.choice(np.arange(len(items)), size=config['tsne_points'], replace=False)
            #keep_random2 = np.random.choice(np.arange(len(users)), size=config['tsne_points'], replace=False)
            #抽取最热门的1000点和最冷门的1000点
            size = int(world.config['tsne_points']/2)
            r1 = np.arange(len(items))
            keep_random1 = np.concatenate((r1[:size], r1[-size:]))
            r2 = np.arange(len(users))
            keep_random2 = np.concatenate((r2[:size], r2[-size:]))
            items = items[keep_random1]
            users = users[keep_random2]
            label_item = label_item[keep_random1]
            label_user = label_user[keep_random2]


            title = ''
            for key in world.LogItems:
                if key != 'comment':
                    title += str(key) + ':' + str(world.config[key]) + '-'
            embs = torch.cat((items, users), dim=0)
            X = TSNE(perplexity=world.config['perplexity'], init='pca', method='barnes_hut').fit_transform(embs)#TODO 其他参数可调
            #X = TSNE(perplexity=config['perplexity']).fit_transform(data)
            
            figsize = (8, 8)
            dpi = 500

            # 创建图形并调整布局
            plt.figure(figsize=figsize, dpi=dpi)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)

            # 绘制散点图，并设置点的大小、颜色映射和标签
            custom_colors_items = ['#D1ECB9','#999999','#999999','#999999','#999999','#999999','#999999','#999999','#999999', '#FFB265']
            custom_colors_items = mcolors.ListedColormap(custom_colors_items)
            custom_colors_users = ['#6DC3F2','#999999','#999999','#999999','#999999','#999999','#999999','#999999','#999999', '#A68BC2']
            custom_colors_users = mcolors.ListedColormap(custom_colors_users)
            classes_item = ['Cold item 1', 'Hot item 10']
            # scatter_item = plt.scatter(X[:len(items), 0], X[:len(items), 1], c=label_item, cmap='coolwarm', s=20, label=classes_item)
            scatter_item = plt.scatter(X[:len(items), 0], X[:len(items), 1], c=label_item, cmap=custom_colors_items, s=20, label=classes_item, zorder=2)
            classes_user = ['Low-degree user 1', 'High-degree user 10']
            # scatter_user = plt.scatter(X[len(items):, 0], X[len(items):, 1], c=label_user, cmap='Pastel2_r', s=20, label=classes_user)
            scatter_user = plt.scatter(X[len(items):, 0], X[len(items):, 1], c=label_user, cmap=custom_colors_users, s=20, label=classes_user, zorder=1)

            # 设置图例字体大小
            plt.rcParams.update({'legend.fontsize': 14})

            # 设置图例位置并添加图例
            legend_item = plt.legend(handles=scatter_item.legend_elements()[0], labels=classes_item, loc='upper left')
            plt.gca().add_artist(legend_item)
            legend_user = plt.legend(handles=scatter_user.legend_elements()[0], labels=classes_user, loc='upper right')
            plt.gca().add_artist(legend_user)

            # 去除x和y轴刻度标签
            plt.xticks([])
            plt.yticks([])

            # 去除图形的外边框
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            # plt.title(title+'\n'+str(world.config['comment'])+'-PCA-Barnes_Hut-Perplexity'+str(world.config['perplexity'])+'@'+str(epoch), fontdict={'weight':'normal','size': 20})
            plt.title("t-SNE Users & Items "+world.config['name']+' @epoch '+str(epoch), fontdict={'weight':'normal','size': 34})
            title += str(world.config['comment'])
            filename = os.path.join(world.FIG_FILE, 't-SNE')
            filename = os.path.join(filename, title)
            if not os.path.exists(filename):
                os.makedirs(filename, exist_ok=True)
            plt.savefig(os.path.join(filename, str(epoch)+'.jpg'))
            #plt.show()
            plt.close()


    def visualize_double_label(self, epoch):
        world.cprint('[double-label]')
        hotGroup = self.precal.popularity.UserPopGroupDict[9]
        hot_user1 = int(hotGroup[111])
        hot_user2 = int(hotGroup[222])
        self.visualize_double_label_for_target(hot_user1, epoch)
        self.visualize_double_label_for_target(hot_user2, epoch)

    def visualize_double_label_for_target(self, target_user, epoch, figsize=(25,25)):
        '''
        target_user : int
        chosen_items : [int]
        '''
        with torch.no_grad():
            # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
            # plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号
            #目标user的全部正样本和随机负样本被t-SNE双标签可视化
            #随机负样本属于各个pop分组的数量相同
            all_pos_for_target = np.concatenate((self.dataset.allPos[target_user], self.dataset.testDict[target_user]))
            random_neg_for_target = set()#其实不一定是负样本
            for group in range(world.config['pop_group']):
                current_group = self.precal.popularity.ItemPopGroupDict[group]#TODO User-->Item
                idx = randint_choice(len(current_group), size=int(0.03*len(current_group)), replace=False)
                random_neg_for_target = random_neg_for_target | set(current_group[idx])
            chosen_items = random_neg_for_target | set(all_pos_for_target)
            chosen_items = np.array(list(chosen_items))

            target_user_emb = self.model.embedding_user.weight.cpu()
            target_user_emb = target_user_emb[target_user]
            chosen_items_emb = self.model.embedding_item.weight.cpu()
            chosen_items_emb = chosen_items_emb[chosen_items]
            embs = torch.cat((target_user_emb.unsqueeze(0), chosen_items_emb), dim=0)
            X = TSNE(perplexity=world.config['perplexity'], init='pca', method='barnes_hut').fit_transform(embs)




            figsize = (8, 8)
            dpi = 500

            # 创建图形并调整布局
            plt.figure(figsize=figsize, dpi=dpi)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)

            classes = ['target user', 'Pos Hot item', 'Pos Cold item', 'Neg Hot item', 'Neg Cold Item']

            pos_item = []
            neg_item = []
            for item in chosen_items:
                if item in all_pos_for_target:
                    pos_item.append(1)
                    neg_item.append(0)
                else:
                    pos_item.append(0)
                    neg_item.append(1)
            pos_item = torch.tensor(pos_item)
            neg_item = torch.tensor(neg_item)

            hot_item = []
            cold_item = []
            for item in chosen_items:
                if int(self.precal.popularity.reverse_ItemPopGroupDict[item]) > 5:#TODO 目前前4组算作热门
                    hot_item.append(1)
                    cold_item.append(0)
                else:
                    hot_item.append(0)
                    cold_item.append(1)
            hot_item = torch.tensor(hot_item)
            cold_item = torch.tensor(cold_item)

            # 绘制散点图，调整点的大小、颜色和透明度
            plt.scatter(X[0, 0], X[0, 1], s=500, c='#D1ECB9', marker='*', alpha=1.0, label=classes[0], edgecolors='#999999', zorder=3)

            index_Pos_Hot_item = torch.cat((torch.tensor([0]), torch.mul(pos_item, hot_item))).to(torch.bool)
            index_Pos_Cold_item = torch.cat((torch.tensor([0]), torch.mul(pos_item, cold_item))).to(torch.bool)
            index_Neg_Hot_item = torch.cat((torch.tensor([0]), torch.mul(neg_item, hot_item))).to(torch.bool)
            index_Neg_Cold_item = torch.cat((torch.tensor([0]), torch.mul(neg_item, cold_item))).to(torch.bool)

            plt.scatter(X[index_Pos_Hot_item, 0], X[index_Pos_Hot_item, 1], s=60, c='#FFB265', marker='o', alpha=1.0, label=classes[1], zorder=2)
            plt.scatter(X[index_Pos_Cold_item, 0], X[index_Pos_Cold_item, 1], s=60, c='#A68BC2', marker='o', alpha=1.0, label=classes[2], zorder=2)
            plt.scatter(X[index_Neg_Hot_item, 0], X[index_Neg_Hot_item, 1], s=40, c='#6DC3F2', marker='o', alpha=1.0, label=classes[3], zorder=1)
            plt.scatter(X[index_Neg_Cold_item, 0], X[index_Neg_Cold_item, 1], s=40, c='#6DC3F2', marker='o', alpha=0.4, label=classes[4], zorder=1)

            # 设置图例字体大小
            plt.rcParams.update({'legend.fontsize': 14})

            # 去除x和y轴刻度标签
            plt.xticks([])
            plt.yticks([])

            # 去除图形的外边框
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            # 显示图例
            plt.legend()

            title = ''
            for key in world.LogItems:
                if key != 'comment':
                    title += str(key) + ':' + str(world.config[key]) + '-'
            # plt.title(title+'\n'+str(world.config['comment'])+'-PCA-Barnes_Hut-Perplexity'+str(world.config['perplexity'])+'@'+str(epoch)+f" for_user: {target_user} ", fontdict={'weight':'normal','size': 20})
            # plt.title("Target User & other items "+world.config['name']+' @epoch '+str(epoch), fontdict={'weight':'normal','size': 34})
            title += str(world.config['comment'])
            filename = os.path.join(world.FIG_FILE, 'double-label')
            filename = os.path.join(filename, title)
            if not os.path.exists(filename):
                os.makedirs(filename, exist_ok=True)
            plt.savefig(os.path.join(filename, str(target_user)+'---'+str(epoch)+'.jpg'))
            #plt.show()
            plt.close()

    def visualize_angle_distribut(self):
        return 0
