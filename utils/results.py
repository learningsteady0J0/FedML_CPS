import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import seaborn as sns

class results_class():
    def __init__(self):
        self.server_results = {'train_losses':[], 'test_losses':[], 'train_acc':[], 'test_acc':[]}
        self.client_idx_results = {}
        self.client_participants_results = {}
        self.data_local_cls_dict = {}

    def get_cls_dict(self, dict):
        self.data_local_cls_dict = dict

    def init_server_results(self, client_num_in_total):
        for i in range(client_num_in_total):
            self.client_idx_results[i] = { 'train_losses':[], 'test_losses':[], 'train_acc':[], 'test_acc':[] }

def arrange_subplots(idx_list, data_list, n_plots):
  """
  ---- Parameters ----
  xs (n_plots, d): list with n_plot different lists of x values that we wish to plot
  ys (n_plots, d): list with n_plot different lists of y values that we wish to plot
  n_plots (int): the number of desired subplots
  """

  # compute the number of rows and columns
  n_cols = int(np.sqrt(n_plots))
  n_rows = int(np.ceil(n_plots / n_cols))

  # setup the plot
  gs = gridspec.GridSpec(n_rows, n_cols)
  scale = max(n_cols, n_rows)
  fig = plt.figure(figsize=(5 * scale, 5 * scale))
  fig.subplots_adjust(hspace=0.5)

  # loop through each subplot and plot values there
  for i in range(n_plots):
    loss_ax = fig.add_subplot(gs[i])
    acc_ax = loss_ax.twinx()

    train_losses = data_list[i]['train_losses']
    train_accs = data_list[i]['train_acc']
    test_accs = data_list[i]['test_acc']

    lns1 = loss_ax.plot(range(1,len(train_losses)+1),train_losses,':',linewidth=5,color='r', label = "train_loss")
    lns2 = acc_ax.plot(range(1,len(train_accs)+1),train_accs,linewidth=5, label = "train_acc")
    lns3 = acc_ax.plot(range(1,len(test_accs)+1),test_accs,linewidth=5, label = "test_acc")

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    acc_ax.set_title("Client {}".format(idx_list[i]), fontsize=15)
    acc_ax.xaxis.set_major_locator(plt.MultipleLocator(len(test_accs)//3))

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    acc_ax.legend(lns, labs, loc=0)


  return fig

def get_data(data_path):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    '''
    print("data list")
    for idx, key in enumerate(data.keys()):
        print("{}. {}".format(idx, key))'''

    return data

def get_round_participants_idx_list(client_participants_results):
    round_participants_idx_list = []
    for round in client_participants_results.keys():
        idx_list = []
        for client_idx in client_participants_results[round].keys():
            idx_list.append(client_idx)
        round_participants_idx_list.append(idx_list)

    return round_participants_idx_list

def get_client_particpants_dict(round_participants_idx_list, total_client_number):
    client_particpants_dict = {}
    total_round = len(round_participants_idx_list)

    for idx in range(total_client_number):
        client_particpants_dict[idx] = np.zeros(total_round, dtype=bool)

    for round, participants_list in enumerate(round_participants_idx_list):
        for participants_idx in  participants_list:
            client_particpants_dict[participants_idx][round] = True

    return client_particpants_dict

def draw_server_results(server_results, dir_path, name="Server"):
    round_num = len(server_results['train_losses'])
    if round_num > 19:
        split_num = round_num // 10
    else:
        split_num = 1

    server_losses = {'train' : server_results['train_losses'], 'test' : server_results['test_losses']}
    server_accs = {'train' : server_results['train_acc'], 'test' : server_results['test_acc']}

    fig, ax = plt.subplots(figsize=(8,4))
    for keys, values in server_losses.items():
        ax.plot(range(1,len(values)+1),values,label = keys)

    ax.set_xlabel('round')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_title("{}_losses".format(name), fontsize=15)
    ax.xaxis.set_major_locator(plt.MultipleLocator(len(values)//split_num))

    fig.savefig(dir_path + '/{}_losses.png'.format(name),dpi=300)

    fig, ax = plt.subplots(figsize=(8,4))
    for keys, values in server_accs.items():
        plt.plot(range(1,len(values)+1),values,label = keys)
    ax.set_xlabel('round')
    ax.set_ylabel('acc')
    ax.legend()
    ax.set_title("{}_acc".format(name), fontsize=15)
    ax.xaxis.set_major_locator(plt.MultipleLocator(len(values)//split_num))

    fig.savefig(dir_path + '/{}_acc.png'.format(name),dpi=300)

def draw_client_results(client_idx_results,client_particpants_dict, dir_path, view_num=0, name="Client"):
    round_num = len(client_idx_results[0]['train_losses'])
    if round_num > 19:
        split_num = round_num // 10
    else:
        split_num = 1
    fig, ax = plt.subplots(figsize=(12,10))
    for idx, client_idx in enumerate(client_idx_results.keys()):
        if view_num == idx:
            break
        plt.plot(range(1,len(client_idx_results[client_idx]['test_acc'])+1),client_idx_results[client_idx]['test_acc'], label = client_idx)
        plt.scatter(np.array(range(1,len(client_idx_results[client_idx]['test_acc'])+1))[client_particpants_dict[client_idx]] \
            , np.array(client_idx_results[client_idx]['test_acc'])[client_particpants_dict[client_idx]])
    ax.set_xlabel('round')
    ax.set_ylabel('acc')
    ax.legend()
    ax.set_title("{}_acc".format(name), fontsize=15)
    ax.xaxis.set_major_locator(plt.MultipleLocator(len(client_idx_results[client_idx]['test_acc'])//1))

    fig.savefig(dir_path + '/{}_acc.png'.format(name),dpi=300)

def draw_client_participants_result(round, participants_client_result, dir_path, n_plots = 3):
    round_participants_client_result = participants_client_result[round]
    if len(round_participants_client_result) < n_plots :
        print("client results is not enough  ## len(client_results) < n_plots ##  ")
        n_plots = len(round_participants_client_result)
        print("drwa_num = {}".format(n_plots))

    test_acc_list = []
    for client_idx in round_participants_client_result.keys():
        test_acc_list.append([client_idx,round_participants_client_result[client_idx]['test_acc'][-1]])
    test_acc_list.sort(key = lambda x:x[1], reverse = True)

    # client_best draw
    top_client_result_idx = []
    top_client_result_data = []
    for idx, acc_list in enumerate(test_acc_list[:n_plots]):
        client_idx = acc_list[0]
        top_client_result_idx.append(client_idx)
        top_client_result_data.append(round_participants_client_result[client_idx])
    top_fig = arrange_subplots(top_client_result_idx, top_client_result_data, n_plots)
    top_fig.savefig(dir_path + 'round{}_top{}.png'.format(round,n_plots))


    worst_client_result_idx = []
    worst_client_result_data = []
    # client_worst draw
    for idx, acc_list in enumerate(test_acc_list[-n_plots:]):
        client_idx = acc_list[0]
        worst_client_result_idx.append(client_idx)
        worst_client_result_idx.reverse()
        worst_client_result_data.append(round_participants_client_result[client_idx])
        worst_client_result_data.reverse()
    worst_fig = arrange_subplots(worst_client_result_idx, worst_client_result_data, n_plots)
    worst_fig.savefig(dir_path + 'round{}_worst{}.png'.format(round,n_plots))

def draw_data_heatmap(train_data_local_cls_dict, dir_path):
    class_list = []
    for Client_number in train_data_local_cls_dict.keys():
        for key in train_data_local_cls_dict[Client_number]['test'].keys():
            if not key in class_list:
                class_list.append(key)
    class_num = len(class_list)

    train_data_dict = {'Client':[], 'label':[], 'value':[]}
    test_data_dict = {'Client':[], 'label':[], 'value':[]}
    total_train_data_dict = {'Client':[0]*class_num, 'label':[ i for i in range(class_num) ], 'value':[0]*class_num}
    total_test_data_dict = {'Client':[0]*class_num, 'label':[ i for i in range(class_num) ], 'value':[0]*class_num}

    for Client_number in train_data_local_cls_dict.keys():
        for label, value in train_data_local_cls_dict[Client_number]['train'].items():
            train_data_dict['Client'].append(Client_number)
            train_data_dict['label'].append(label)
            train_data_dict['value'].append(int(value))
            total_train_data_dict['value'][label] += int(value)
        for label, value in train_data_local_cls_dict[Client_number]['test'].items():
            test_data_dict['Client'].append(Client_number)
            test_data_dict['label'].append(label)
            test_data_dict['value'].append(int(value))
            total_test_data_dict['value'][label] += int(value)



    pd_data = pd.DataFrame(train_data_dict)
    test_pd_data = pd.DataFrame(test_data_dict)
    total_pd_data = pd.DataFrame(total_train_data_dict)

    df = pd.pivot_table(pd_data, index='Client', columns='label', values='value',fill_value = 0)
    test_df = pd.pivot_table(test_pd_data, index='Client', columns='label', values='value',fill_value = 0)
    total_df = pd.pivot_table(total_pd_data, index='Client', columns='label', values='value',fill_value = 0)

    df = df.iloc[:20,:].astype(int)
    test_df = test_df.iloc[:20,:].astype(int)

    fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.heatmap(df,  cmap='YlGnBu', annot=False, fmt='d')
    ax.set_title('Train_data', fontsize=20)
    fig.savefig(dir_path + '/traindata_heatmap.png',dpi=300)

    fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.heatmap(test_df,  cmap='YlGnBu', annot=False, fmt='d')
    ax.set_title('Test_data', fontsize=20)
    fig.savefig(dir_path + '/testdata_heatmap.png',dpi=300)

    total_fig, total_ax = plt.subplots(figsize=(8,4))
    total_ax = sns.heatmap(total_df,  cmap='YlGnBu', annot=True, fmt='d')
    total_ax.set_title('Heatmap of Flight by seaborn', fontsize=20)
    total_fig.savefig(dir_path + '/total_data_heatmap.png',dpi=300)


def draw_results(args):
    if args.aa == True:
        pass

    if args.aa == True:
        pass

    if args.aa == True:
        pass

def get_top_worst_idx(round, participants_client_result, n_plot=3):
        round_participants_client_result = participants_client_result[round]
        if len(round_participants_client_result) < n_plots :
            print("client results is not enough  ## len(client_results) < n_plots ##  ")
            n_plots = len(round_participants_client_result)
            print("drwa_num = {}".format(n_plots))

        test_acc_list = []
        for client_idx in round_participants_client_result.keys():
            test_acc_list.append([client_idx,round_participants_client_result[client_idx]['test_acc'][-1]])
        test_acc_list.sort(key = lambda x:x[1], reverse = True)

        top_client_result_idx = []
        for idx, acc_list in enumerate(test_acc_list[:n_plots]):
            client_idx = acc_list[0]
            top_client_result_idx.append(client_idx)

        worst_client_result_idx = []
        for idx, acc_list in enumerate(test_acc_list[-n_plots:]):
            client_idx = acc_list[0]
            worst_client_result_idx.append(client_idx)
            worst_client_result_idx.reverse()

        return top_client_result_idx, worst_client_result_idx

if __name__ == "__main__":
    # data_list[0] : client_results
    # data_list[1] : server_results
    exp_name = "exp1$fedAVG_mnist_test"
    dir_path = "../results/{}/".format(exp_name)
    data = get_data(dir_path + "results.pickle")

    server_results = data['server_results']
    client_idx_results = data['client_idx_results']
    client_participants_results = data['client_participants_results']
    train_data_local_cls_dict = data['data_local_cls_dict']
    total_client_number = len(client_idx_results.keys())
    round_participants_idx_list = get_round_participants_idx_list(client_participants_results)
    client_particpants_dict = get_client_particpants_dict(round_participants_idx_list,total_client_number)

    server_losses = {'train' : server_results['train_losses'], 'test' : server_results['test_losses']}
    server_accs = {'train' : server_results['train_acc'], 'test' : server_results['test_acc']}
    
    draw_server_results(server_results, dir_path, name='server')

    draw_client_participants_result(0, client_participants_results, dir_path, n_plots =3)

    draw_client_results(client_idx_results,client_particpants_dict, dir_path, 10, name="Client")

    draw_data_heatmap(train_data_local_cls_dict, dir_path)
