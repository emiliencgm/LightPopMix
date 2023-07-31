'''
1
Long-tail 10 groups
'''
import numpy as np
import matplotlib.pyplot as plt

X_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X1 = np.array([1.3, 2.4, 2.5, 6.0, 2.0, 1.3, 2.4, 2.5, 6.0, 2.0])
X2 = np.array([2.3, 2.7, 4.5, 7.0, 2.4, 2.3, 2.7, 4.5, 7.0, 2.4])
X3 = np.array([1.8, 2.1, 4.0, 5.0, 1.4, 1.8, 2.1, 4.0, 5.0, 1.4])
X4 = np.array([1.8, 2.1, 4.0, 5.0, 1.4, 1.8, 2.1, 4.0, 5.0, 1.4])
X5 = np.array([1.3, 2.4, 2.5, 6.0, 2.0, 1.3, 2.4, 2.5, 6.0, 2.0])
X6 = np.array([2.3, 2.7, 4.5, 7.0, 2.4, 2.3, 2.7, 4.5, 7.0, 2.4])

def groups(X_values, X1, X2, X3, X4, X5, X6):

    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 500

    # 绘制图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95)

    bar_width = 0.1
    bar_positions_X1 = X_values - 2.5*bar_width
    bar_positions_X2 = X_values -1.5*bar_width
    bar_positions_X3 = X_values - 0.5*bar_width
    bar_positions_X4 = X_values + 0.5*bar_width
    bar_positions_X5 = X_values + 1.5*bar_width
    bar_positions_X6 = X_values + 2.5*bar_width
    '''
    red:#DB7B6E
    orange:#FFB265
    yellow:#FFF2CC
    green:#D1ECB9
    blue:#6DC3F2
    purple:#A68BC2
    '''
    ax.bar(bar_positions_X1, X1, width=bar_width, color='#DB7B6E', label='LightGCN', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X2, X2, width=bar_width, color='#FFB265', label='SGL', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X3, X3, width=bar_width, color='#FFF2CC', label='SimGCL', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X4, X4, width=bar_width, color='#D1ECB9', label='PDA', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X5, X5, width=bar_width, color='#6DC3F2', label='BC loss', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X6, X6, width=bar_width, color='#A68BC2', label='Ours', edgecolor='black', linewidth=0.5)


    # 添加标签和标题
    plt.xlabel('Group ID', fontsize=14, fontweight='bold')
    plt.ylabel('$Recall^{(g)}@20$', fontsize=16, fontweight='bold')
    plt.title('Yelp2018', fontsize=14)
    plt.xticks(X_values)
    plt.legend(loc='upper left', fontsize=14)

    # 设置坐标轴刻度字体大小和加粗
    plt.tick_params(axis='both', labelsize=12, width=2, length=6, labelcolor='black')

    # 增加横向的网格参考线
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # 显示图形
    plt.savefig('tt.jpg')

groups(X_values, X1, X2, X3, X4, X5, X6)









'''
2
hyperparameters  折线图
双纵轴：Overall和Long-Tail
'''
X_values = [1, 2, 3, 4, 5]
Y_values_1 = [10, 15, 8, 12, 20]
Y_values_2 = [50, 30, 25, 40, 45]

def double_y(X_values, Y_values_1,Y_values_2):
    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 450

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.15, bottom=0.07, right=0.85, top=0.93)

    # title = ('τ')
    # plt.title(title,fontsize=16)
    plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    plt.tick_params(axis='both',labelsize=14)
    plot1 = ax1.plot(X_values, Y_values_1, color='#DB7B6E', label='$Recall@20$', marker='o', linestyle='-')
    ax1.set_ylabel('Overall  $Recall@20$', fontsize = 16, color='black', labelpad=10)
    # ax1.set_ylim(0,240)
    for tl in ax1.get_yticklabels():
        tl.set_color('#DB7B6E')    
    ax2 = ax1.twinx()
    plot2 = ax2.plot(X_values, Y_values_2, color='#A68BC2', label='$Recall^{(1)}@20$', marker='s', linestyle='--')
    ax2.set_ylabel('Long-tail  $Recall^{(1)}$',fontsize=16, color='black', labelpad=12)
    # ax2.set_ylim(0,0.08)
    ax2.tick_params(axis='y',labelsize=14)
    for tl in ax2.get_yticklabels():
        tl.set_color('#A68BC2')                    
    # ax2.set_xlim(1966,2014.15)
    lines = plot1 + plot2           
    ax1.legend(lines,[l.get_label() for l in lines], loc='upper center', fontsize=14)    
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0],ax1.get_ybound()[1],9)) 
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0],ax2.get_ybound()[1],9)) 
    for ax in [ax1,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)          

    # 显示图形
    plt.savefig('tt.jpg')


double_y(X_values, Y_values_1,Y_values_2)



'''
？
projectors:
双纵轴：Overall和Long-Tail

或改成表格？
或用训练中的结果？——其实不用单画图，用WandB组合出来即可
'''



