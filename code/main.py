"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import dataloader
import precalcul
import world
from world import cprint
from world import cprint_rare
import model
import loss
import procedure
import torch
from os.path import join
import time
import visual
from pprint import pprint
import utils
import wandb
import math
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torch_geometric import seed_everything


def plot_MLP(epoch, precal, total_loss):
    '''
    plot every 3 epochs
    '''
    with torch.no_grad():
        if epoch % 3 == 0:
            max_pop_i = precal.popularity.max_pop_i
            pop_i = np.arange(1, max_pop_i, 10)
            max_pop_i = math.log(max_pop_i)
            centroid = np.arange(0, 1, 0.01)

            input_mlp_batch = []
            for i in pop_i:
                a = [0.]*(5+2*0)
                a[1] = math.log(i)
                # a[1] = i
                input_mlp_batch.append(a)
            input_mlp_batch = torch.Tensor(input_mlp_batch).to(world.device)
            output_mlp_batch = total_loss.MLP_model(input_mlp_batch)
            output_mlp_batch = torch.arccos(torch.clamp(output_mlp_batch,-1+1e-7,1-1e-7))
            output_mlp_batch = np.array(output_mlp_batch.cpu())
            plt1 = plt.plot(pop_i, output_mlp_batch)
            plt.xlabel("pop_i")
            plt.ylabel("theta_ui = arccos()")
            plt.savefig("my_plot_pop_item.png")
            wandb.log({"MLP(pop_item)": wandb.Image("my_plot_pop_item.png", caption="epoch:{}".format(epoch))})
            plt.clf()

            input_mlp_batch = []
            for i in centroid:
                a = [0.]*(5+2*0)
                a[2] = i
                input_mlp_batch.append(a)
            input_mlp_batch = torch.Tensor(input_mlp_batch).to(world.device)
            output_mlp_batch = total_loss.MLP_model(input_mlp_batch)
            output_mlp_batch = torch.arccos(torch.clamp(output_mlp_batch,-1+1e-7,1-1e-7))
            output_mlp_batch = np.array(output_mlp_batch.cpu())
            plt2 = plt.plot(centroid, output_mlp_batch)
            plt.xlabel("centroid")
            plt.ylabel("theta_ui = arccos()")
            plt.savefig("my_plot_centroid.png")
            wandb.log({"MLP(centroid)": wandb.Image("my_plot_centroid.png", caption="epoch:{}".format(epoch))})
            plt.clf()

            output_mlp_batch = total_loss.batch_weight
            output_mlp_batch = torch.arccos(torch.clamp(output_mlp_batch,-1+1e-7,1-1e-7))
            n, bins, patches = plt.hist(x=np.array(output_mlp_batch.cpu()), bins=50, density=False)
            for i in range(len(n)):
                plt.text(bins[i], n[i]*1.00, int(n[i]), fontsize=6, horizontalalignment="center")
            plt.xlabel("theta_ui = arccos()")
            plt.ylabel("num")
            plt.savefig("my_plot_hist.png")
            wandb.log({"Hist(pos_weight)": wandb.Image("my_plot_hist.png", caption="epoch:{}".format(epoch))})
            plt.clf()

def grouped_recall(epoch, result):
    current_best_recall_group = np.zeros((world.config['pop_group'], len(world.config['topks'])))
    for i in range(len(world.config['topks'])):
        k = world.config['topks'][i]
        for group in range(world.config['pop_group']):
            current_best_recall_group[group, i] = result['recall_pop_Contribute'][group][i]
    return current_best_recall_group

def main():
    print('DEVICE:',world.device, world.args.cuda)
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(world.device))

    project = world.config['project']
    name = world.config['name']
    tag = world.config['tag']
    notes = world.config['notes']
    group = world.config['group']
    job_type = world.config['job_type']
    # os.environ['WANDB_MODE'] = 'dryrun'#TODO WandB上传 
    wandb.init(project=project, name=name, tags=tag, group=group, job_type=job_type, config=world.config, save_code=True, sync_tensorboard=False, notes=notes)
    wandb.define_metric("epoch")
    wandb.define_metric(f"{world.config['dataset']}"+'/loss', step_metric='epoch')
    for k in world.config['topks']:
        wandb.define_metric(f"{world.config['dataset']}"+f'/recall@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/ndcg@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/precision@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_recall@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_ndcg@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_precision@{str(k)}', step_metric='epoch')
        for group in range(world.config['pop_group']):
            wandb.define_metric(f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}", step_metric='epoch')
    wandb.define_metric(f"{world.config['dataset']}"+f"/training_time", step_metric='epoch')
    
    wandb.define_metric(f"{world.config['dataset']}"+'/pop_classifier_acc', step_metric='epoch')

    for group in range(world.config['pop_group']):
        wandb.define_metric(f"{world.config['dataset']}"+f"/groups/Rating_group_{group+1}", step_metric='epoch')


    world.make_print_to_file()

    # utils.set_seed(world.config['seed'])
    seed_everything(seed=world.config['seed'])

    print('==========config==========')
    pprint(world.config)
    print('==========config==========')

    cprint('[DATALOADER--START]')
    datasetpath = join(world.DATA_PATH, world.config['dataset'])
    dataset = dataloader.dataset(world.config, datasetpath)
    cprint('[DATALOADER--END]')

    cprint('[PRECALCULATE--START]')
    start = time.time()
    precal = precalcul.precalculate(world.config, dataset)
    end = time.time()
    print('precal cost : ',end-start)
    cprint('[PRECALCULATE--END]')

    cprint('[SAMPLER--START]')
    sampler = precalcul.sampler(dataset=dataset, precal=precal)
    cprint('[SAMPLER--END]')
    

    models = {'LightGCN':model.LightGCN}
    Recmodel = models[world.config['model']](world.config, dataset, precal).to(world.device)

    classifier = model.Classifier(input_dim=world.config['latent_dim_rec'], out_dim=world.config['pop_group'], precal=precal)

    augmentation = None
    

    losss = {'Adaptive':loss.Adaptive_loss}
    total_loss = losss[world.config['loss']](world.config, Recmodel, precal)

    train = procedure.Train(total_loss)
    test = procedure.Test()
    
    emb_optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
    emb_optimizer.add_param_group({'params':total_loss.MLP_model.parameters()})
    pop_optimizer = torch.optim.Adam(classifier.parameters(), lr=world.config['lr'])
    optimizer = {'emb':emb_optimizer, 'pop':pop_optimizer}

    quantify = visual.Quantify(dataset, Recmodel, precal)


    try:
        best_result_recall = 0.
        best_result_ndcg = 0.
        stopping_step = 0
        best_result_recall_group = None
        if world.config['if_valid']:
            best_valid_recall = 0.
            stopping_valid_step = 0


        for epoch in range(world.config['epochs']):
            wandb.log({"epoch": epoch})
            start = time.time()
            #====================VISUAL====================
            if world.config['if_visual'] == 1 and epoch % world.config['visual_epoch'] == 0:
                cprint("[Visualization]")
                if world.config['if_tsne'] == 1:
                    quantify.visualize_tsne(epoch)
                if world.config['if_double_label'] == 1:
                    quantify.visualize_double_label(epoch)
            #====================AUGMENT====================
            #None
            #====================TRAIN====================                    
            cprint('[TRAIN]')
            start_train = time.time()
            avg_loss, avg_pop_acc = train.train(sampler, Recmodel, epoch, optimizer, classifier)
            end_train = time.time()
            wandb.log({ f"{world.config['dataset']}"+'/loss': avg_loss})
            wandb.log({f"{world.config['dataset']}"+f"/training_time": end_train - start_train})

            wandb.log({ f"{world.config['dataset']}"+'/pop_classifier_acc': avg_pop_acc})

            if epoch % 1== 0:
                #====================VALID====================
                if world.config['if_valid']:
                    cprint("[valid]")
                    result = test.valid(dataset, Recmodel, multicore=world.config['if_multicore'])
                    if result["recall"][0] > best_valid_recall:#默认按照@20的效果early stop
                        stopping_valid_step = 0
                        advance = (result["recall"][0] - best_valid_recall)
                        best_valid_recall = result["recall"][0]
                        # print("find a better model")
                        cprint_rare("find a better valid recall", str(best_valid_recall), extra='++'+str(advance))
                        wandb.run.summary['best valid recall'] = best_valid_recall  
                    else:
                        stopping_valid_step += 1
                        if stopping_valid_step >= world.config['early_stop_steps']:
                            print(f"early stop triggerd at epoch {epoch}, best valid recall: {best_valid_recall}")
                            #将当前参数配置和获得的最佳结果记录
                            break
                    for i in range(len(world.config['topks'])):
                        k = world.config['topks'][i]
                        wandb.log({ f"{world.config['dataset']}"+f'/valid_recall@{str(k)}': result["recall"][i],
                                    f"{world.config['dataset']}"+f'/valid_ndcg@{str(k)}': result["ndcg"][i],
                                    f"{world.config['dataset']}"+f'/valid_precision@{str(k)}': result["precision"][i]})
                        
                #====================TEST====================
                cprint("[TEST]")
                result = test.test(dataset, Recmodel, precal, epoch, world.config['if_multicore'])
                if result["recall"][0] > best_result_recall:#默认按照@20的效果early stop
                    stopping_step = 0
                    advance = (result["recall"][0] - best_result_recall)
                    best_result_recall = result["recall"][0]
                    # print("find a better model")
                    cprint_rare("find a better recall", str(best_result_recall), extra='++'+str(advance))
                    best_result_recall_group = grouped_recall(epoch, result)
                    wandb.run.summary['best test recall'] = best_result_recall  

                    # if world.config['if_visual'] == 1:
                    #     cprint("[Visualization]")
                    #     if world.config['if_tsne'] == 1:
                    #         quantify.visualize_tsne(epoch)
                    #     if world.config['if_double_label'] == 1:
                    #         quantify.visualize_double_label(epoch)

                    #torch.save(Recmodel.state_dict(), weight_file)
                else:
                    stopping_step += 1
                    if stopping_step >= world.config['early_stop_steps']:
                        print(f"early stop triggerd at epoch {epoch}, best recall: {best_result_recall}, in group: {best_result_recall_group}")
                        #将当前参数配置和获得的最佳结果记录
                        break
                
                if world.config['if_visual'] == 1:
                    Ratings_group = Recmodel.getItemRating()
                    for group in range(world.config['pop_group']):
                        wandb.log({f"{world.config['dataset']}"+f"/groups/Rating_group_{group+1}": Ratings_group[group]})


                for i in range(len(world.config['topks'])):
                    k = world.config['topks'][i]
                    wandb.log({ f"{world.config['dataset']}"+f'/recall@{str(k)}': result["recall"][i],
                                f"{world.config['dataset']}"+f'/ndcg@{str(k)}': result["ndcg"][i],
                                f"{world.config['dataset']}"+f'/precision@{str(k)}': result["precision"][i]})
                    for group in range(world.config['pop_group']):
                        wandb.log({f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}": result['recall_pop_Contribute'][group][i]})

            during = time.time() - start
            print(f"total time cost of epoch {epoch}: ", during)

            if world.config['loss'] in ['Adaptive', 'simple_Adaptive']:
                #plot MLP(pop)
                # plot_MLP(epoch, precal, total_loss)TODO
                pass
                

    finally:
        cprint(world.config['c'])
        wandb.finish()
        cprint(world.config['c'])


if __name__ == '__main__':
    main()