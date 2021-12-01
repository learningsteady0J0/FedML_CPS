import pickle
import matplotlib.pyplot as plt

def get_data():
    pass

def draw_losses(train_losses, test_losses, save_dir, name=""):
    plt.plot(range(1,len(train_losses)+1),train_losses,'r',c = 'b',label = 'train loss')
    plt.plot(range(1,len(test_losses)+1),test_losses,'r',c = 'r',label = 'test loss')
    plt.legend()
    plt.savefig(save_dir + '/{}_losses.png'.format(name),dpi=300)
    plt.cla()

def draw_acc(train_acc, test_acc, save_dir, name=""):
    plt.plot(range(1,len(train_acc)+1),train_acc,'r',c = 'b',label = 'train acc')
    plt.plot(range(1,len(test_acc)+1),test_acc,'r',c = 'r',label = 'test acc')
    plt.legend()
    plt.savefig(save_dir + '/{}_acc.png'.format(name),dpi=300)
    plt.cla()


# todo list
# 1. use subplot
def draw_clientresult(client_results, round, save_dir, draw_num = 3):
    round_client_results = client_results[round]
    if len(round_client_results) < draw_num * 2:
        print("client results is not enough  ## len(client_results) < draw_num * 2 ##  ")
        draw_num = len(round_client_results) // 2
        print("drwa_num = {}".format(draw_num))

    best_acc_list = []
    for idx, client_result in enumerate(round_client_results):
        best_acc_list.append([idx,client_result['best_acc']])
    best_acc_list.sort(key = lambda x:x[1], reverse = True)

    # client_best draw
    for idx, acc_list in enumerate(best_acc_list[:draw_num]):
        client_result_idx = acc_list[0]
        draw_acc(round_client_results[client_result_idx]['train_acc'], round_client_results[client_result_idx]['test_acc'], save_dir, 'round{}_client_best{}'.format(round,idx+1))
        draw_losses(round_client_results[client_result_idx]['train_losses'], round_client_results[client_result_idx]['test_losses'], save_dir, 'round{}_client_best{}'.format(round,idx+1))

    # client_worst draw
    for idx, acc_list in enumerate(best_acc_list[-draw_num:]):
        client_result_idx = acc_list[0]
        draw_acc(round_client_results[client_result_idx]['train_acc'], round_client_results[client_result_idx]['test_acc'], save_dir, 'round{}_client_worst{}'.format(round,draw_num-idx))
        draw_losses(round_client_results[client_result_idx]['train_losses'], round_client_results[client_result_idx]['test_losses'], save_dir, 'round{}_client_worst{}'.format(round,draw_num-idx))

    # full draw
    for idx, acc_list in enumerate(best_acc_list[-draw_num:]):
        client_result_idx = acc_list[0]
        test_acc = round_client_results[client_result_idx]['test_acc']
        plt.plot(range(1,len(test_acc)+1),test_acc, label = '{}'.format(idx+1))
    plt.legend()
    plt.savefig(save_dir + '/full_acc.png',dpi=300)
    plt.cla()
