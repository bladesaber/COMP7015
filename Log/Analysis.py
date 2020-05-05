import numpy as np
import matplotlib.pyplot as plt
import os
import math

def move_avg(data, weight):
    move_line = []
    move_line.append(data[0])
    for i in data[1:]:
        move_line.append(move_line[-1]*(1-weight)+i*weight)
    return move_line

def plot_figure(figure_name, dataSet, names):
    plt.figure(figure_name)
    plt.yticks(np.arange(0.0,7.0,0.05))
    for i, data in enumerate(dataSet):
        plt.plot(range(len(data)), data,label=names[i], linewidth=1.5)
        plt.axhline(y=data[-1], ls="--", linewidth=0.7)
        # plt.text(0, data[-1], round(data[-1],3), fontsize=10)
    plt.legend()

def run():
    #if person computer
    project_dir = 'D:\GoogleDrive\Colab Notebooks\HKBU_AI_Classs\COMP7015_Mini_Project/Log/npy_dir'
    # if colab
    # project_dir = 'drive/My Drive/Colab Notebooks/HKBU_AI_Classs/COMP7015_Mini_Project/Log'

    mini_cnn = np.load(os.path.join(project_dir, 'mini_cnn.npy'), allow_pickle=True).item()
    mini_resnet = np.load(os.path.join(project_dir, 'mini_resnet.npy'), allow_pickle=True).item()
    mini_inception = np.load(os.path.join(project_dir, 'mini_inception.npy'), allow_pickle=True).item()

    mini_cnn_zcore = np.load(os.path.join(project_dir, 'mini_cnn_zcore.npy'), allow_pickle=True).item()
    mini_resnet_zcore = np.load(os.path.join(project_dir, 'mini_resnet_zcore.npy'), allow_pickle=True).item()
    mini_inception_zcore = np.load(os.path.join(project_dir, 'mini_inception_zcore.npy'), allow_pickle=True).item()

    mini_cnn_augu = np.load(os.path.join(project_dir, 'mini_cnn_augu_zcore.npy'), allow_pickle=True).item()
    mini_resnet_augu = np.load(os.path.join(project_dir, 'mini_resnet_augu_zcore.npy'), allow_pickle=True).item()
    mini_inception_zero_augu = np.load(os.path.join(project_dir, 'mini_inception_augu_zcore.npy'), allow_pickle=True).item()
    mini_inception_augu = np.load(os.path.join(project_dir, 'mini_inception_augu.npy'),allow_pickle=True).item()

    # ------------------------------------------------------------------
    cnn_train_loss = mini_cnn['train_cost']
    inception_train_loss = mini_inception['train_cost']
    resnet_train_loss = mini_resnet['train_cost']

    cnn_train_loss_zcore = mini_cnn_zcore['train_cost']
    resnet_train_loss_zcore = mini_resnet_zcore['train_cost']
    inception_train_loss_zcore = mini_inception_zcore['train_cost']

    cnn_train_loss_augu = mini_cnn_augu['train_cost']
    inception_train_loss_augu_zero = mini_inception_zero_augu['train_cost']
    resnet_train_loss_augu = mini_resnet_augu['train_cost']
    inception_train_loss_augu = mini_inception_augu['train_cost']

    # ---------------------------------------------------------------------
    cnn_train_acc = mini_cnn['train_acc']
    inception_train_acc = mini_inception['train_acc']
    resnet_train_acc = mini_resnet['train_acc']

    cnn_train_acc_zcore = mini_cnn_zcore['train_acc']
    resnet_train_acc_zcore = mini_resnet_zcore['train_acc']
    inception_train_acc_zcore = mini_inception_zcore['train_acc']

    cnn_train_acc_augu = mini_cnn_augu['train_acc']
    inception_train_acc_augu_zero = mini_inception_zero_augu['train_acc']
    resnet_train_acc_augu = mini_resnet_augu['train_acc']
    inception_train_acc_augu = mini_inception_augu['train_acc']

    # --------------------------------------------------------------------
    cnn_test_loss = mini_cnn['test_cost']
    inception_test_loss = mini_inception['test_cost']
    resnet_test_loss = mini_resnet['test_cost']

    cnn_test_loss_zcore = mini_cnn_zcore['test_cost']
    resnet_test_loss_zcore = mini_resnet_zcore['test_cost']
    inception_test_loss_zcore = mini_inception_zcore['test_cost']

    cnn_test_loss_augu = mini_cnn_augu['test_cost']
    inception_test_loss_augu_zero = mini_inception_zero_augu['test_cost']
    resnet_test_loss_augu = mini_resnet_augu['test_cost']
    inception_test_loss_augu = mini_inception_augu['test_cost']

    # --------------------------------------------------------------------
    cnn_test_acc = mini_cnn['test_acc']
    inception_test_acc = mini_inception['test_acc']
    resnet_test_acc = mini_resnet['test_acc']

    cnn_test_acc_zcore = mini_cnn_zcore['test_acc']
    resnet_test_acc_zcore = mini_resnet_zcore['test_acc']
    inception_test_acc_zcore = mini_inception_zcore['test_acc']

    cnn_test_acc_augu = mini_cnn_augu['test_acc']
    inception_test_acc_augu_zero = mini_inception_zero_augu['test_acc']
    resnet_test_acc_augu = mini_resnet_augu['test_acc']
    inception_test_acc_augu = mini_inception_augu['test_acc']

    # --------------------------------------------------------------------
    cnn_train_time = mini_cnn['time']
    inception_train_time = mini_inception['time']
    resnet_train_time = mini_resnet['time']

    # --------------------------------------------------------------------
    # plot_figure(figure_name='train_valid_loss',
    #             dataSet=[
    #                 # cnn_train_loss,
    #                 # inception_train_loss,
    #                 # resnet_train_loss,
    #                 cnn_train_loss_zcore,
    #                 inception_train_loss_zcore,
    #                 resnet_train_loss_zcore,
    #             ],
    #             names=[
    #                 # 'mini_cnn_train_cost',
    #                 # 'mini_inception_train_cost',
    #                 # 'mini_resnet_train_cost',
    #                 'mini_cnn_train_loss_zcore',
    #                 'mini_inception_train_loss_zcore',
    #                 'mini_resnet_train_loss_zcore',
    #             ])

    # plot_figure(figure_name='test_valid_acc',
    #             dataSet=[
    #                 # cnn_test_acc,
    #                 # inception_test_acc,
    #                 # resnet_test_acc,
    #                 # cnn_test_acc_zcore,
    #                 # inception_test_acc_zcore,
    #                 # resnet_test_acc_zcore,
    #                 cnn_test_acc_augu,
    #                 # inception_test_acc_augu,
    #                 # resnet_test_acc_augu,
    #             ],
    #             names=[
    #                 # 'mini_cnn_test_accracy',
    #                 # 'mini_inception_test_accuracy',
    #                 # 'mini_resnet_test_accuracy',
    #                 # 'mini_cnn_test_acc_zcore',
    #                 # 'mini_inception_test_acc_zcore',
    #                 # 'mini_resnet_test_acc_zcore',
    #                 'mini_cnn_test_acc_augmentation',
    #                 # 'inception_test_acc_augu',
    #                 # 'resnet_test_acc_augu',
    #             ])

    step1 = math.ceil(len(cnn_train_acc_zcore)/len(cnn_test_acc_zcore))
    step2 = math.ceil(len(inception_train_acc)/len(inception_test_acc))
    step3 = math.ceil(len(resnet_train_acc_zcore)/len(resnet_test_acc_zcore))
    step4 = math.ceil(len(cnn_train_acc_augu)/len(cnn_test_acc_augu))
    plot_figure(figure_name='test_train',
                dataSet=[
                    # cnn_train_acc_zcore[np.arange(0, len(cnn_train_acc_zcore),step1)],
                    # inception_train_acc[np.arange(0, len(inception_train_acc), step2)],
                    # resnet_train_acc_zcore[np.arange(0, len(resnet_train_acc_zcore), step3)],
                    cnn_test_acc_zcore,
                    # inception_test_acc,
                    # resnet_test_acc_zcore,
                    cnn_test_acc_augu,
                    # inception_test_acc_augu,
                    # resnet_test_acc_augu,
                    # cnn_train_acc_augu[np.arange(0, len(cnn_train_acc_augu),step4)],
                ],
                names=[
                    # 'mini_cnn_train_acc',
                    # 'mini_inception_train_acc',
                    # 'mini_resnet_train_acc',
                    'mini_cnn_test_acc',
                    # 'mini_inception_test_acc',
                    # 'mini_resnet_test_acc',
                    'mini_cnn_test_acc_augmentation',
                    # 'mini_inception_test_acc_augmentation',
                    # 'mini_resnet_test_acc_augmentation',
                    # 'mini_cnn_train_acc_augmentation',
                ])

    # plot_figure(figure_name='train_zcore_loss',
    #             dataSet=[cnn_train_loss_zcore, inception_train_loss, resnet_train_loss_zcore, inception_train_loss_zcore],
    #             names=['cnn_train_loss_zcore', 'inception_train_loss', 'resnet_train_loss_zcore', 'inception_train_loss_zcore'])

    # plot_figure(figure_name='train_zcore_augu_loss',
    #             dataSet=[cnn_train_loss_augu, inception_train_loss_augu, resnet_train_loss_augu, inception_train_loss_augu_zero],
    #             names=['cnn_train_loss_augu', 'inception_train_loss_augu', 'resnet_train_loss_augu', 'inception_train_loss_augu_zero'])

    # plot_figure(figure_name='test_valid_loss',
    #             dataSet=[cnn_test_loss, inception_test_loss, resnet_test_loss],
    #             names=['cnn_test_loss', 'inception_test_loss', 'resnet_test_loss'])

    # plot_figure(figure_name='test_zcore_loss',
    #             dataSet=[cnn_test_loss_zcore, inception_test_loss_zcore, resnet_test_loss_zcore],
    #             names=['cnn_test_loss_zcore', 'inception_test_loss_zcore', 'resnet_test_loss_zcore'])

    # plot_figure(figure_name='augu_test_loss',
    #             dataSet=[cnn_test_loss_augu, resnet_test_loss_augu, inception_test_loss_augu, inception_test_loss_augu_zero],
    #             names=['cnn_test_loss_augu', 'resnet_test_loss_augu', 'inception_test_loss_augu', 'inception_test_loss_augu_zero'])

    # plot_figure(figure_name='train_zero_accuracy',
    #             dataSet=[cnn_train_acc_zcore, inception_train_acc_zcore, resnet_train_acc_zcore],
    #             names=['cnn_train_acc', 'inception_train_acc', 'resnet_train_acc'])

    # plot_figure(figure_name='test_zero_accuracy',
    #             dataSet=[
    #                 move_avg(cnn_test_acc_zcore,0.05),
    #                 move_avg(inception_test_acc_zcore, 0.05),
    #                 move_avg(resnet_test_acc_zcore, 0.05)],
    #             names=['cnn_test_acc', 'inception_test_acc', 'resnet_test_acc'])

    # ------------------------------------------------------------------------
    # # zcore 对传统 CNN 有效， 效果明显
    # plot_figure(figure_name='cnn_compare',
    #             dataSet=[cnn_train_loss, cnn_train_loss_zcore],
    #             names=['cnn_train_loss', 'cnn_train_loss_zcore'])
    # # zcore 对传统 CNN 有效， 效果不明显
    # plot_figure(figure_name='inception_compare',
    #             dataSet=[inception_train_loss, inception_train_loss_zcore],
    #             names=['inception_train_loss', 'inception_train_loss_zcore'])
    # # zcore 对传统 CNN 有效， 效果不明显
    # plot_figure(figure_name='resnet_compare',
    #             dataSet=[resnet_train_loss, resnet_train_loss_zcore],
    #             names=['resnet_train_loss', 'resnet_train_loss_zcore'])

    # plot_figure(figure_name='zcore_loss_converge',
    #             dataSet=[
    #                 np.abs((cnn_train_loss_zcore[np.arange(0, len(cnn_train_loss_zcore), 20)])/64.-cnn_test_loss_zcore[:-1]/2048.),
    #                 np.abs((inception_train_loss_zcore[np.arange(0, len(inception_train_loss_zcore), 20)]) / 64. - inception_test_loss_zcore / 2048.),
    #                 np.abs((resnet_train_loss_zcore[np.arange(0, len(resnet_train_loss_zcore), 20)]) / 64. - resnet_test_loss_zcore / 2048.),
    #             ],
    #             names=['cnn_converge', 'inception_converge', 'resnet_converge'])

    # plot_figure(figure_name='zcore_acc_converge',
    #             dataSet=[
    #                 # cnn_train_acc_zcore[np.arange(0, len(cnn_train_acc_zcore), 20)], cnn_test_acc_zcore[np.arange(0, len(cnn_test_acc_zcore), 2)],
    #                 # inception_train_acc_zcore[np.arange(0, len(inception_train_acc_zcore), 20)], inception_test_acc_zcore[np.arange(0, len(inception_test_acc_zcore), 2)],
    #                 resnet_train_acc_zcore[np.arange(0, len(resnet_train_acc_zcore), 20)], resnet_test_acc_zcore[np.arange(0, len(resnet_test_acc_zcore), 2)],
    #             ],
    #             names=[
    #                 # 'cnn_train_acc', 'cnn_test_acc'
    #                 # 'inception_train_acc', 'inception_test_acc'
    #                 'resnet_train_acc', 'resnet_test_acc'
    #                 ])

    # -----------------------------------------------------------------------------------------------
    # plot_figure(figure_name='augu_loss_compare',
    #             dataSet=[
    #                 resnet_train_loss_zcore, resnet_train_loss_augu,
    #                 inception_train_loss_zcore, inception_train_loss_augu
    #             ],
    #             names=[
    #                 'resnet_train_loss_zcore', 'resnet_train_loss_augu',
    #                 'inception_train_loss_zcore', 'inception_train_loss_augu'
    #             ])

    # plot_figure(figure_name='augu_acc_converge',
    #             dataSet=[
    #                 inception_train_acc_augu[np.arange(0, len(inception_train_acc_augu), 20)], inception_test_acc_augu,
    #                 # resnet_train_acc_augu[np.arange(0, len(resnet_train_acc_augu), 20)], resnet_test_acc_augu,
    #             ],
    #             names=[
    #                 'inception_train_acc', 'inception_test_acc'
    #                 # 'resnet_train_acc', 'resnet_test_acc'
    #                 ])

    # plot_figure(figure_name='augu_train_loss',
    #             dataSet=[cnn_train_loss_augu, resnet_train_loss_augu, inception_train_loss_augu],
    #             names=['cnn_train_loss_augu', 'resnet_train_loss_augu', 'inception_train_loss_augu'])

    # ------------------------------------------------------------------------------------------
    # plot_figure(figure_name='test_acc',
    #             dataSet=[cnn_test_acc_augu, resnet_test_acc_augu, inception_test_acc_augu_zero, inception_test_acc_augu],
    #             names=['cnn_test_acc_augu', 'resnet_test_acc_augu', 'inception_test_acc_augu_zero', 'inception_test_acc_augu'])

    plt.show()

def run1():
    #if person computer
    project_dir = 'D:\GoogleDrive\Colab Notebooks\HKBU_AI_Classs\COMP7015_Mini_Project/Log/npy_dir'
    # if colab
    # project_dir = 'drive/My Drive/Colab Notebooks/HKBU_AI_Classs/COMP7015_Mini_Project/Log'

    mini_cnn_zcore = np.load(os.path.join(project_dir, 'mini_cnn_zcore.npy'), allow_pickle=True).item()
    mini_cnn_augu = np.load(os.path.join(project_dir, 'mini_cnn_augu_zcore.npy'), allow_pickle=True).item()
    cnn_augu = np.load(os.path.join(project_dir, 'cnn_augu_zcore.npy'), allow_pickle=True).item()
    cnn_com_augu = np.load(os.path.join(project_dir, 'cnn_com_augu_zcore.npy'), allow_pickle=True).item()

    mini_resnet_zcore = np.load(os.path.join(project_dir, 'mini_resnet_zcore.npy'), allow_pickle=True).item()
    mini_resnet_augu = np.load(os.path.join(project_dir, 'mini_resnet_augu_zcore.npy'), allow_pickle=True).item()
    resnet_augu = np.load(os.path.join(project_dir, 'resnet_augu_zcore.npy'), allow_pickle=True).item()
    resnet_adjust_augu = np.load(os.path.join(project_dir, 'resnet_adjust_augu_zcore.npy'), allow_pickle=True).item()
    resnet_e_augu = np.load(os.path.join(project_dir, 'resnet_e_augu_zcore.npy'), allow_pickle=True).item()
    resnet_augu_improve_v1 = np.load(os.path.join(project_dir, 'resnet_improved_v1_augu_zcore.npy'), allow_pickle=True).item()

    mini_inception_augu = np.load(os.path.join(project_dir, 'mini_inception_augu_zcore.npy'), allow_pickle=True).item()
    inception_augu = np.load(os.path.join(project_dir, 'inception_augu_zcore.npy'), allow_pickle=True).item()
    inception_augu_improve_v1 = np.load(os.path.join(project_dir, 'inception_improved_v1_augu_zcore.npy'), allow_pickle=True).item()

    # --------------------------------------------------------------------
    mini_cnn_train_loss_zcore = mini_cnn_zcore['train_cost']
    mini_cnn_train_loss_augu = mini_cnn_augu['train_cost']
    cnn_train_loss_augu = cnn_augu['train_cost']
    cnn_com_train_loss_augu = cnn_com_augu['train_cost']

    mini_resnet_train_loss_zcore = mini_resnet_zcore['train_cost']
    mini_resnet_train_loss_augu = mini_resnet_augu['train_cost']
    resnet_train_loss_augu = resnet_augu['train_cost']
    resnet_train_loss_augu_adjust = resnet_adjust_augu['train_cost']
    resnet_train_loss_augu_e = resnet_e_augu['train_cost']
    resnet_train_loss_augu_improve_v1 = resnet_augu_improve_v1['train_cost']

    mini_inception_train_loss_augu = mini_inception_augu['train_cost']
    inception_train_loss_augu = inception_augu['train_cost']
    inception_train_loss_augu_improve_v1 = inception_augu_improve_v1['train_cost']

    # ---------------------------------------------------------------------
    mini_cnn_train_acc_zcore = mini_cnn_zcore['train_acc']
    mini_cnn_train_acc_augu = mini_cnn_augu['train_acc']
    cnn_train_acc_augu = cnn_augu['train_acc']
    cnn_com_train_acc_augu = cnn_com_augu['train_acc']

    mini_resnet_train_acc_zcore = mini_resnet_zcore['train_acc']
    mini_resnet_train_acc_augu = mini_resnet_augu['train_acc']
    resnet_train_acc_augu = resnet_augu['train_acc']
    resnet_train_acc_augu_adjust = resnet_adjust_augu['train_acc']
    resnet_train_acc_augu_e = resnet_e_augu['train_acc']
    resnet_train_acc_augu_improve_v1 = resnet_augu_improve_v1['train_acc']

    mini_inception_train_acc_augu = mini_inception_augu['train_acc']
    inception_train_acc_augu = inception_augu['train_acc']
    inception_train_acc_augu_improve_v1 = inception_augu_improve_v1['train_acc']

    # --------------------------------------------------------------------
    mini_cnn_test_loss_zcore = mini_cnn_zcore['test_cost']
    mini_cnn_test_loss_augu = mini_cnn_augu['test_cost']
    cnn_test_loss_augu = cnn_augu['test_cost']
    cnn_com_test_loss_augu = cnn_com_augu['test_cost']

    mini_resnet_test_loss_zcore = mini_resnet_zcore['test_cost']
    mini_resnet_test_loss_augu = mini_resnet_augu['test_cost']
    resnet_test_loss_augu = resnet_augu['test_cost']
    resnet_test_loss_augu_adjust = resnet_adjust_augu['test_cost']
    resnet_test_loss_augu_e = resnet_e_augu['test_cost']
    resnet_test_loss_augu_improve_v1 = resnet_augu_improve_v1['test_cost']

    mini_inception_test_loss_augu = mini_inception_augu['test_cost']
    inception_test_loss_augu = inception_augu['test_cost']
    inception_test_loss_augu_improve_v1 = inception_augu_improve_v1['test_cost']

    # --------------------------------------------------------------------
    mini_cnn_test_acc_zcore = mini_cnn_zcore['test_acc']
    mini_cnn_test_acc_augu = mini_cnn_augu['test_acc']
    cnn_test_acc_augu = cnn_augu['test_acc']
    cnn_com_test_acc_augu = cnn_com_augu['test_acc']

    mini_resnet_test_acc_zcore = mini_resnet_zcore['test_acc']
    mini_resnet_test_acc_augu = mini_resnet_augu['test_acc']
    resnet_test_acc_augu = resnet_augu['test_acc']
    resnet_test_acc_augu_adjust = resnet_adjust_augu['test_acc']
    resnet_test_acc_augu_e = resnet_e_augu['test_acc']
    resnet_test_acc_augu_improve_v1 = resnet_augu_improve_v1['test_acc']

    mini_inception_test_acc_augu = mini_inception_augu['test_acc']
    inception_test_acc_augu = inception_augu['test_acc']
    inception_test_acc_augu_improve_v1 = inception_augu_improve_v1['test_acc']

    # --------------------------------------------------------------------
    # plot_figure(figure_name='train_cost',
    #             dataSet=[
    #                 mini_cnn_train_loss_zcore,
    #                 mini_cnn_train_loss_augu,
    #                 cnn_train_loss_augu],
    #             names=[
    #                 'mini_cnn_train_loss_zcore',
    #                 'mini_cnn_train_loss_augu',
    #                 'cnn_train_loss_augu'])

    # plot_figure(figure_name='acc',
    #             dataSet=[
    #                 mini_cnn_test_acc_zcore[np.arange(0, len(mini_cnn_test_acc_zcore), 2)],
    #                 mini_cnn_test_acc_augu,
    #                 cnn_test_acc_augu
    #             ],
    #             names=['mini_cnn_test_acc_zcore', 'mini_cnn_test_acc_augu', 'cnn_test_acc_augu'])

    # plot_figure(figure_name='train_cost',
    #             dataSet=[
    #                 cnn_train_loss_augu,
    #                 resnet_train_loss_augu,
    #                 mini_resnet_train_loss_augu,
    #                 mini_resnet_train_loss_zcore
    #             ],
    #             names=[
    #                 'cnn_train_loss_augu',
    #                 'resnet_train_loss_augu',
    #                 'mini_resnet_train_loss_augu',
    #                 'mini_resnet_train_loss_zcore'
    #             ])

    # plot_figure(figure_name='test_acc',
    #             dataSet=[
    #                 cnn_test_acc_augu,
    #                 resnet_test_acc_augu,
    #                 mini_resnet_test_acc_augu,
    #                 mini_resnet_test_acc_zcore[np.arange(0, len(mini_resnet_test_acc_zcore), 2)]
    #             ],
    #             names=[
    #                 'cnn_test_acc_augu',
    #                 'resnet_test_acc_augu',
    #                 'mini_resnet_test_acc_augu',
    #                 'mini_resnet_test_acc_zcore'
    #             ])

    # plot_figure(figure_name='train_loss',
    #             dataSet=[
    #                 # mini_cnn_train_loss_augu,
    #                 # mini_resnet_train_loss_augu,
    #                 # mini_inception_train_loss_augu,
    #                 # cnn_train_loss_augu,
    #                 # inception_train_loss_augu,
    #                 resnet_train_loss_augu,
    #                 cnn_com_train_loss_augu,
    #                 # resnet_train_loss_augu_adjust,
    #                 # resnet_train_loss_augu_e,
    #                 # resnet_train_loss_augu_improve_v1,
    #                 # inception_train_loss_augu_improve_v1,
    #             ],
    #             names=[
    #                 # 'mini_cnn_train_loss_augu',
    #                 # 'mini_resnet_train_loss_augu',
    #                 # 'mini_inception_train_loss_augu',
    #                 # 'cnn_train_loss_augu',
    #                 # 'inception_train_loss_augu',
    #                 'resnet_train_loss_augu',
    #                 'cnn_com_train_loss_augu',
    #                 # 'resnet_train_loss_augu_adjust',
    #                 # 'resnet_train_loss_augu_e',
    #                 # 'resnet_train_loss_augu_improve_v1',
    #                 # 'inception_train_loss_augu_improve_v1',
    #             ])

    plot_figure(figure_name='test_acc',
                dataSet=[
                    mini_cnn_test_acc_augu,
                    mini_resnet_test_acc_augu,
                    mini_inception_test_acc_augu,
                    # cnn_test_acc_augu,
                    inception_test_acc_augu,
                    cnn_com_test_acc_augu,
                    # resnet_test_acc_augu,
                    # resnet_test_acc_augu_adjust,
                    # resnet_test_acc_augu_e,
                    resnet_test_acc_augu_improve_v1,
                    # inception_test_acc_augu_improve_v1,
                ],
                names=[
                    'mini_cnn_test_acc',
                    'mini_resnet_test_acc',
                    'mini_inception_test_acc',
                    # 'cnn_test_acc_augu',
                    'deeper_inception_test_acc',
                    'deeper_cnn_test_acc',
                    # 'resnet_test_acc_augu',
                    # 'resnet_test_acc_augu_adjust',
                    # 'resnet_test_acc_augu_e',
                    'deeper_resnet_test_acc',
                    # 'inception_test_acc_augu_improve_v1',
                ])

    plt.show()

if __name__ == '__main__':
    # run()
    run1()
