import tensorflow as tf
import os,tools,math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def loss_fun(out_net,label_in_one_fixed_shape):
    out_reshaped = tf.reshape(out_net,(-1,7,7,30))
    loss1 = loss2 = loss3 = loss4 = loss5 = 0
    for k in range(0,label_in_one_fixed_shape.shape[0]):
        stick_30_cell_and_label = []
        cell_loc_alone = []
        cell_no_obj = []
        for i in range(0, 56):
            if int(label_in_one_fixed_shape[k][i][4]) != 1:
                cell_loc = tools.cell_cal(label_in_one_fixed_shape[k][i][0:4])
                stick_30_lable = label_in_one_fixed_shape[k][i]
                cell_and_label = [cell_loc,stick_30_lable,i]
                if cell_loc not in cell_loc_alone:
                    cell_loc_alone.append(cell_loc)
                    stick_30_cell_and_label.append(cell_and_label)
            else:
                cell_no_obj.append(tools.cell_cal(label_in_one_fixed_shape[k][i][0:4]))
        for cell_and_label in stick_30_cell_and_label:
            cell_loc = cell_and_label[0]
            iou1 = tools.iou(out_reshaped[k][cell_loc[0]][cell_loc[1]][22:26], cell_and_label[1][0:4], cell_loc)
            iou2 = tools.iou(out_reshaped[k][cell_loc[0]][cell_loc[1]][26:30], cell_and_label[1][0:4], cell_loc)
            if iou1 > iou2:
                x_cent = int((cell_and_label[1][0] + cell_and_label[1][2]) / 2)
                y_cent = int((cell_and_label[1][1] + cell_and_label[1][3]) / 2)
                x_logit = (x_cent - cell_loc[0] * (448 / 7)) / (448 / 7)
                y_logit = (y_cent - cell_loc[1] * (448 / 7)) / (448 / 7)
                loss1 = loss1 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][22] - x_logit),2) + \
                        pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][23] - y_logit),2)
                w = (cell_and_label[1][2] - cell_and_label[1][0]) / 448
                h = (cell_and_label[1][3] - cell_and_label[1][1]) / 448
                loss2 = loss2 + pow((math.sqrt(w) - out_reshaped[k][cell_loc[0]][cell_loc[1]][24]),2) + \
                        pow((math.sqrt(h) - out_reshaped[k][cell_loc[0]][cell_loc[1]][25]), 2)
                loss3 = loss3 + pow((1 - out_reshaped[k][cell_loc[0]][cell_loc[1]][20]),2)
                loss4 = loss4 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][21]),2)
                loss5 = loss5 + pow((out_reshaped[k][int(cell_loc[0])][int(cell_loc[1])][int(cell_and_label[1][4])-1] - 1),2)#标签的分类值是从1开始的
                for aa in range(0,20):
                    if aa != (cell_and_label[1][4]-1):
                        loss5 = loss5 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][aa]),2)
            else:#iou2 >= iou1
                x_cent = int((cell_and_label[1][0] + cell_and_label[1][2]) / 2)
                y_cent = int((cell_and_label[1][1] + cell_and_label[1][3]) / 2)
                x_logit = (x_cent - cell_loc[0] * (448 / 7)) / (448 / 7)
                y_logit = (y_cent - cell_loc[1] * (448 / 7)) / (448 / 7)
                loss1 = loss1 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][26] - x_logit), 2) + \
                        pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][27] - y_logit), 2)
                w = (cell_and_label[1][2] - cell_and_label[1][0]) / 448
                h = (cell_and_label[1][3] - cell_and_label[1][1]) / 448
                loss2 = loss2 + pow((math.sqrt(w) - out_reshaped[k][cell_loc[0]][cell_loc[1]][28]), 2) + \
                        pow((math.sqrt(h) - out_reshaped[k][cell_loc[0]][cell_loc[1]][29]), 2)
                loss3 = loss3 + pow((1 - out_reshaped[k][cell_loc[0]][cell_loc[1]][21]), 2)
                loss4 = loss4 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][20]), 2)
                loss5 = loss5 + pow((out_reshaped[k][int(cell_loc[0])][int(cell_loc[1])][int(cell_and_label[1][4]) - 1] - 1),2)  # 标签的分类值是从1开始的
                for c in range(0, 20):
                    if c != cell_and_label[1][4] - 1:
                        loss5 = loss5 + pow((out_reshaped[k][cell_loc[0]][cell_loc[1]][c]), 2)
            for cell in cell_no_obj:
                loss4 = loss4 + pow((out_reshaped[k][cell[0]][cell[1]][20]), 2) + \
                        pow((out_reshaped[k][cell[0]][cell[1]][21]), 2)
    return 5 * loss1 + 5 * loss2 + loss3 + 0.5 * loss4 + loss5
