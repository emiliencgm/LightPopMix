"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go SGL")
    #WandB
    parser.add_argument('--project', type=str, default='project', help="wandb project")
    parser.add_argument('--name', type=str, default='name', help="wandb name")   
    parser.add_argument('--notes', type=str, default='testing in main.py', help="wandb notes")   
    parser.add_argument('--tag', nargs='+', default='test' ,help='wandb tags')
    parser.add_argument('--group', type=str, default='Ours', help="wandb group") 
    parser.add_argument('--job_type', type=str, default='yelp2018', help="wandb job_type") 
    #Hyperparameters===========================================================================================================================================
    parser.add_argument('--temp_tau', type=float, default=0.1, help="tau in InfoNCEloss")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay == lambda2")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1 == coef of Contrstloss")
    #===========================================================================================================================================
    parser.add_argument('--early_stop_steps', type=int, default=30, help="early stop steps")
    parser.add_argument('--latent_dim_rec', type=int, default=64, help="latent dim for rec")
    parser.add_argument('--num_layers', type=int, default=3, help="num layers of LightGCN")
    parser.add_argument('--epochs', type=int, default=500, help="training epochs")
    parser.add_argument('--if_multicore', type=int, default=1, help="whether use multicores in Test")
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size in BPR_Contrast_Train")    
    parser.add_argument('--topks', nargs='?', default='[20, 40]', help="topks [@20, @40] for test")
    parser.add_argument('--test_u_batch_size', type=int, default=2048, help="users batch size for test")
    parser.add_argument('--pop_group', type=int, default=10, help="Num of groups of Popularity")
    parser.add_argument('--cuda', type=str, default='0', help="cuda id")
    parser.add_argument('--visual_epoch', type=int, default=1, help="visualize every tsne_epoch")
    parser.add_argument('--if_double_label', type=int, default=1, help="whether use item categories label along with popularity group")
    parser.add_argument('--if_tsne', type=int, default=1, help="whether use t-SNE")
    parser.add_argument('--tsne_group', nargs='?', default='[0, 9]', help="groups [0, 9] for t-SNE")    
    parser.add_argument('--tsne_points', type=int, default=2000, help="Num of points of users/items in t-SNE")
    parser.add_argument('--if_visual', type=int, default=0, help="whether use visualization, i.e. t_sne, double_label")  
    #Architecture===========================================================================================================================================
    parser.add_argument('--model', type=str, default='LightGCN', help="Now available:LightGCN")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset:[yelp2018,  gawalla, ifashion, amazon-book, last-fm, MIND]") 
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--loss', type=str, default='Adaptive', help="loss function: Adaptive")
    parser.add_argument('--augment', type=str, default='No', help="Augmentation: No, Adaptive, Learner")    
    parser.add_argument('--centroid_mode', type=str, default='eigenvector', help="Centroid mode: degree, pagerank, eigenvector")
    parser.add_argument('--commonNeighbor_mode', type=str, default='SC', help="Common Neighbor mode: JS, SC, CN, LHN")
    parser.add_argument('--adaptive_method', type=str, default='mlp', help="Adaptive coef method: centroid, commonNeighbor, homophily, mlp")
    parser.add_argument('--init_method', type=str, default='Normal', help="UI embeddings init method: Xavier or Normal")
    parser.add_argument('--perplexity', type=int, default=50, help="perplexity for T-SNE")
    parser.add_argument('--comment', type=str, default='pyg_implementation', help="comment for the experiment")
    parser.add_argument('--if_valid', type=int, default=0, help="whether use validtion set")
    parser.add_argument('--sampling', type=str, default='uii', help="sampling method")
    parser.add_argument('--projector', type=str, default='w', help="with or without projectors")
    #===========================================================================================================================================
    parser.add_argument('--c', type=str, default='testing', help="note something for this experiment")


    return parser.parse_args()