import os

import torch
from nets.unet_training import CE_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,cls_weights, num_classes, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        #----------------------#
        #   Forward propagation
        #----------------------#
        outputs,output_s2,output_vv,output_temp,ws2,ws1= model_train(imgs)  #final,final_s2,final_vv,final_temp
        L_w=[0.25,0.25,0.25,0.25]
        #----------------------#
        #   Loss Calculation
        #----------------------#
        loss = CE_Loss(outputs, pngs, weights, ws2, num_classes = num_classes)*L_w[0]
        loss_s2 = CE_Loss(output_s2, pngs, weights, ws2, num_classes = num_classes)*L_w[1]
        loss_vv = CE_Loss(output_vv, pngs, weights, ws1, num_classes = num_classes)*L_w[2]
        loss_temp=CE_Loss(output_temp,pngs, weights, ws1, num_classes = num_classes)*L_w[3]
        loss = loss+loss_s2+loss_vv+loss_temp


        with torch.no_grad():
            #-------------------------------#
            #   Calculate f-score
            #-------------------------------#
            _f_score = f_score(outputs, labels)*L_w[0]
            _f_score_s2 = f_score(output_s2, labels)*L_w[1]
            _f_score_vv = f_score(output_vv, labels)*L_w[2]
            _f_score_temp = f_score(output_temp, labels)*L_w[3]
            _f_score = _f_score_temp+_f_score_s2+_f_score_vv+_f_score

        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   Forward propagation
            #----------------------#
            outputs,output_s2,output_vv,output_temp,ws2,ws1 = model_train(imgs)  ##final,final_s2,final_vv,final_temp
            L_w=[0.25,0.25,0.25,0.25]
            #----------------------#
            #   Loss Calculation
            #----------------------#

            loss = CE_Loss(outputs, pngs, weights, ws2, num_classes = num_classes)*L_w[0]
            loss_s2 = CE_Loss(output_s2, pngs, weights, ws2, num_classes = num_classes)*L_w[1]
            loss_vv = CE_Loss(output_vv, pngs, weights, ws1, num_classes = num_classes)*L_w[2]
            loss_temp=CE_Loss(output_temp,pngs, weights, ws1, num_classes = num_classes)*L_w[3]
            loss=loss+loss_s2+loss_vv+loss_temp


            #-------------------------------#
            #   Calculate f-score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)*L_w[0]
            _f_score_s2 = f_score(output_s2, labels)*L_w[1]
            _f_score_vv = f_score(output_vv, labels)*L_w[2]
            _f_score_temp = f_score(output_temp, labels)*L_w[3]
            _f_score = _f_score_temp+_f_score_s2+_f_score_vv+_f_score

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   Save weight values
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

