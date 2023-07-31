import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='LightPopMix', help="method: LightPopMix, LightPopMix_wo_p, LightPopMix_lambda, LightPopMix_tau, LightPopMix_grid")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--valid', type=int, default=0, help="validation")
    parser.add_argument('--c', type=str, default='noting to comment', help="comment in CMD after training")
    return parser.parse_args()
args = parse_args()
if args.valid == 1:
    project = 'TestCode_Valid'
else:
    project = 'TestCode_No_Valid'

#For all
lr = 0.001
weight_decay = 1e-4
num_layers = 3
latent_dim_rec = 64
batch_size =2048
init_method = 'Normal'


if args.method == 'LightPopMix':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag LightPopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag LightPopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')




elif args.method == 'LightPopMix_tau':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm']:
        lambda1 = 0.1
        centroid = 'eigenvector'
        for temp_tau in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
            os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_tau --tag Ablation_tau --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                        --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion']:
        lambda1 = 0.1
        centroid = 'pagerank'
        for ltemp_tau in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
            os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_tau --tag Ablation_tau --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                        --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')






elif args.method == 'LightPopMix_lambda':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm']:
        temp_tau = 0.1
        centroid = 'eigenvector'
        for lambda1 in [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5, 1.]:
            os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_lambda --tag Ablation_lambda --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                        --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion']:
        temp_tau = 0.1
        centroid = 'pagerank'
        for lambda1 in [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5, 1.]:
            os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_lambda --tag Ablation_lambda --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                        --sampling uii --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')





elif args.method == 'LightPopMix_wo_p':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_wo_projectors --tag Ablation_proj --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --projector wo --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')
    
    if args.dataset in ['ifashion']:
        temp_tau = 0.1
        lambda1 = 0.1
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes Ablation_wo_projectors --tag Ablation_proj --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --projector wo --dataset {args.dataset} --cuda {args.device} --comment _ --if_valid {args.valid} --if_visual {args.visual} --visual_epoch 5 --seed {args.seed} --c {args.c}')



else:
    print('#=====================#')
    print(args.method)
    print('#=====================#')
