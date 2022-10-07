from tqdm import tqdm
import torch

from evaluation import calc_accuracy, get_confusion_matrix_image, get_mean_squared_error
from evaluation import get_sample_dict, update_hardsample_indice, draw_cam
from evaluation import write_age_hard_sample
import torchvision.utils as vutils
from utils import tensor_rgb2bgr



def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_mse,
    loss_ce,
    optimizer,
    scheduler,
    meta, 
    writer = None,
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        
        train(ep, max_epoch, model, train_loader, loss_mse, loss_ce, optimizer, writer, print_every)
        if scheduler is not None:
            scheduler.step()
        


        if ep % test_every == 0:
            acc = test(ep, max_epoch, model, test_loader, writer)
            
            writer.update_by_acc(model, acc)

       
        
        if ep == 1 or ep % save_every == 0:
            writer.save(model, ep)
            
    writer.close()
    

def tester(
    model,
    test_loader,
    writer,
    visualizer,
    confusion_matrix,
    csv = False,
    hard_sample = False,
    age_hard_sample = False,
    age_ratio_thres = 10,
    age_diff_thres = 0.2    
):

    pbar=tqdm(total=len(test_loader))
    print('Dataset length: {}'.format(len(test_loader)))
    acc = test(
        None,None,
        model, test_loader, writer,
        confusion_matrix = confusion_matrix,
        csv = csv,
        hard_sample = hard_sample,
        age_hard_sample = age_hard_sample,
        age_ratio_thres = age_ratio_thres,
        age_diff_thres = age_diff_thres   
    )
    
    writer.close()




def train(ep, max_epoch, model, train_loader, loss_mse, loss_ce, optimizer, writer, _print_every):
    model.train()

    epoch_error = 0.0
    total_loss = 0.0
    mse_loss = 0.0
    ce_loss = 0.0

    print_every = len(train_loader) // _print_every     
    if print_every == 0:
        print_every = 1

    score_dict = {
        'pred_age' : 0,
        'gt_age' : 0,

        'pred_cac' : [],
        'gt_cac' : [],

        'pred_sex' : [],
        'gt_sex' : [],

        'pred_age2' : 0,
        'gt_age2' : 0,

        'pred_cac2' : [],
        'gt_cac2' : [],

        'pred_sex2' : [],
        'gt_sex2' : [],
    }

    step = 0
    step_cnt = 1

    global_step = (ep - 1) * len(train_loader)
    local_step = 0


    tb_dict = {
        'age' : 0,
        'sex' : 0,
        'cac' : 0,
        'recon' : 0,
        'age2' : 0,
        'cac2' : 0,
        
    }


    for i, batch in enumerate(train_loader):
        image = batch['image'].cuda()


        # sex
        gt_age = batch['age'].cuda()
        gt_age2 = batch['age2'].cuda()

        # age
        gt_sex = batch['sex'].cuda()
        gt_sex2 = batch['sex2'].cuda()

        # cac
        gt_cac = batch['cac'].cuda()
        gt_cac2 = batch['cac2'].cuda()


        output_dict = model(image)


        # recon loss
        recon_loss = loss_mse(output_dict['x_hat'], image)

        
        # age loss
        age_loss1 = loss_mse(output_dict['pred']['age'][:, 0:1], gt_age)
        age_loss2 = loss_mse(output_dict['pred']['age'][:, 1:2], gt_age2)
        age_loss = age_loss1 + age_loss2
        
        # sex loss
        sex_loss = loss_ce(output_dict['pred']['sex'][:, 0:2], gt_sex) + loss_ce(output_dict['pred']['sex'][:, 2:4], gt_sex2)
        
        # cac loss (TODO: empty csv value check)
        if True:
            cac_loss1 = loss_ce(output_dict['pred']['cac'][:, 0:5], gt_cac)
            cac_loss2 = loss_ce(output_dict['pred']['cac'][:, 5:10], gt_cac2)
            cac_loss = cac_loss1 + cac_loss2

        else:
            cac_loss = 0.0
        

        #! TODO : loss weight check
        total_loss = 0.3 * recon_loss + 0.5 * age_loss + 0.1 * sex_loss + cac_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # classification summary
        score_dict['pred_sex'].append(output_dict['pred']['sex'][:, 0:2].detach().cpu())
        score_dict['pred_sex2'].append(output_dict['pred']['sex'][:, 2:4].detach().cpu())
        score_dict['gt_sex'].append(gt_sex.detach().cpu())
        score_dict['gt_sex2'].append(gt_sex2.detach().cpu())

        score_dict['pred_cac'].append(output_dict['pred']['cac'][:, 0:5].detach().cpu())
        score_dict['pred_cac2'].append(output_dict['pred']['cac'][:, 5:10].detach().cpu())
        score_dict['gt_cac'].append(gt_cac.detach().cpu())
        score_dict['gt_cac2'].append(gt_cac2.detach().cpu())




        # regression summary
        score_dict['pred_age'] += age_loss1.item() 
        score_dict['pred_age2'] += age_loss2.item() 




        tb_dict['age'] += age_loss1.item()
        tb_dict['age2'] += age_loss2.item()

        tb_dict['sex'] += sex_loss.item()

        tb_dict['cac'] += cac_loss1.item()
        tb_dict['cac2'] += cac_loss2.item()

        tb_dict['recon'] += recon_loss.item()



        step += 1
        global_step += 1
        local_step += 1


        if (i+1) % print_every == 0:
            summary_message = 'Epoch [{}/{}] Step[{}/{}] '.format(ep, max_epoch, step_cnt, _print_every)

            for k, v in tb_dict.items():
                title_name = 'train/loss-{}'.format(k)
                writer.add_scalar(title_name, v, global_step)
                tb_dict[k] = 0.0

                summary_message += '\t{}-loss'.format(k) + '{:.4f}'.format(v)

            print(summary_message)        

            step = 0
            step_cnt += 1

    # classification summary (accuracy)

    cac_preds = torch.cat(score_dict['pred_cac'])
    cac_preds2 = torch.cat(score_dict['pred_cac2'])
    cac_gt    = torch.cat(score_dict['gt_cac'])
    cac_gt2    = torch.cat(score_dict['gt_cac2'])

    cac_acc = torch.mean((cac_preds.argmax(dim=1) == cac_gt).float()) 
    cac_acc2 = torch.mean((cac_preds2.argmax(dim=1) == cac_gt2).float()) 

    sex_preds = torch.cat(score_dict['pred_sex'])
    sex_preds2 = torch.cat(score_dict['pred_sex2'])
    sex_gt    = torch.cat(score_dict['gt_sex'])
    sex_gt2    = torch.cat(score_dict['gt_sex2'])

    sex_acc = torch.mean((sex_preds.argmax(dim=1) == sex_gt).float()) 
    sex_acc2 = torch.mean((sex_preds2.argmax(dim=1) == sex_gt2).float())   



    # regression summary (error)

    age_error = score_dict['pred_age']
    age_error2 = score_dict['pred_age2']

    summary_message = 'Epoch [{}/{}] '.format(ep, max_epoch)
    summary_message += '\tcac-acc1: {:.4f}'.format(cac_acc)
    summary_message += '\tcac-acc2: {:.4f}'.format(cac_acc2)
    summary_message += '\tsex-acc1: {:.4f}'.format(sex_acc)
    summary_message += '\tsex-acc2: {:.4f}'.format(sex_acc2)
    summary_message += '\tage-error: {:.4f}'.format(age_error)
    summary_message += '\tage-error2: {:.4f}'.format(age_error2)

    print (summary_message)

    writer.add_scalar('train/cac-acc1', cac_acc, ep)
    writer.add_scalar('train/cac-acc2', cac_acc2, ep)

    writer.add_scalar('train/sex-acc1', sex_acc, ep)
    writer.add_scalar('train/sex-acc2', sex_acc2, ep)

    writer.add_scalar('train/age-error1', age_error, ep)
    writer.add_scalar('train/age-error2', age_error2, ep)



@torch.no_grad() # stop calculating gradient

def test(
    ep, 
    max_epoch, 
    model, 
    test_loader, 
    writer, 
    confusion_matrix = False, 
    csv = False, 
    hard_sample = False,
    age_hard_sample = False,
    age_diff_thres = None,
    age_ratio_thres = None 
):


    model.eval()


    score_dict = {
        'pred_age' : 0,
        'gt_age' : 0,

        'pred_cac' : [],
        'gt_cac' : [],

        'pred_sex' : [],
        'gt_sex' : [],

        'pred_age2' : 0,
        'gt_age2' : 0,

        'pred_cac2' : [],
        'gt_cac2' : [],

        'pred_sex2' : [],
        'gt_sex2' : [],
    }

    step = 0
    step_cnt = 1
    local_step = 0


    tb_dict = {
        'age' : 0,
        'sex' : 0,
        'cac' : 0,
        'recon' : 0,
        'age2' : 0,
        'cac2' : 0,
        
    }


    for i, batch in enumerate(test_loader):
        image = batch['image'].cuda()


        # age
        gt_actual_age = batch['actual_age']
        gt_actual_age2 = batch['actual_age2']


        # sex
        gt_sex = batch['sex']
        gt_sex2 = batch['sex2']

        # cac
        gt_cac = batch['cac']
        gt_cac2 = batch['cac2']


        output_dict = model(image)

        # classification (cac,sex)
        score_dict['pred_cac'].append(output_dict['pred']['cac'][:, 0:5].detach().cpu())
        score_dict['gt_cac'].append(gt_cac)
        score_dict['pred_cac2'].append(output_dict['pred']['cac'][:, 5:10].detach().cpu())
        score_dict['gt_cac2'].append(gt_cac2)

        score_dict['pred_sex'].append(output_dict['pred']['sex'][:, 0:2].detach().cpu())
        score_dict['gt_sex'].append(gt_sex)
        score_dict['pred_sex2'].append(output_dict['pred']['sex'][:, 2:4].detach().cpu())
        score_dict['gt_sex2'].append(gt_sex2)


        # regression (age)
        # TODO : implement
        '''
        gt_actual_age
        (output_dict['pred']['age'][:, 0:1] * 60.0) + 20

        gt_actual_age2
        (output_dict['pred']['age'][:, 1:2] * 60.0) + 20
        



        step += 1
        global_step += 1
        local_step += 1
        '''



    # classification summary (accuracy)
    cac_preds = torch.cat(score_dict['pred_cac'])
    cac_preds2 = torch.cat(score_dict['pred_cac2'])
    cac_gt    = torch.cat(score_dict['gt_cac'])
    cac_gt2    = torch.cat(score_dict['gt_cac2'])

    cac_acc = torch.mean((cac_preds.argmax(dim=1) == cac_gt).float()) 
    cac_acc2 = torch.mean((cac_preds2.argmax(dim=1) == cac_gt2).float()) 

    sex_preds = torch.cat(score_dict['pred_sex'])
    sex_preds2 = torch.cat(score_dict['pred_sex2'])
    sex_gt    = torch.cat(score_dict['gt_sex'])
    sex_gt2    = torch.cat(score_dict['gt_sex2'])

    sex_acc = torch.mean((sex_preds.argmax(dim=1) == sex_gt).float()) 
    sex_acc2 = torch.mean((sex_preds2.argmax(dim=1) == sex_gt2).float())   



    # regression summary (error)
    # TODO : implement
    age_error = 0.0
    age_error2 = 0.0


    summary_message = 'Test Summary Epoch [{}/{}] '.format(ep, max_epoch)
    summary_message += '\tcac-acc1: {:.4f}'.format(cac_acc)
    summary_message += '\tcac-acc2: {:.4f}'.format(cac_acc2)
    summary_message += '\tsex-acc1: {:.4f}'.format(sex_acc)
    summary_message += '\tsex-acc2: {:.4f}'.format(sex_acc2)
    summary_message += '\tage-error: {:.4f}'.format(age_error)
    summary_message += '\tage-error2: {:.4f}'.format(age_error2)

    print (summary_message)

    writer.add_scalar('test/cac-acc1', cac_acc, ep)
    writer.add_scalar('test/cac-acc2', cac_acc2, ep)

    writer.add_scalar('test/sex-acc1', sex_acc, ep)
    writer.add_scalar('test/sex-acc2', sex_acc2, ep)

    writer.add_scalar('test/age-error1', age_error, ep)
    writer.add_scalar('test/age-error2', age_error2, ep)

    return (cac_acc + cac_acc2)/2


def grad_cam(model, data_loader, writer, cam, export_csv, n_class, task_type):
    model.eval()
    pbar = tqdm(total=len(data_loader))

    print ('Dataset length: {}'.format(len(data_loader)))

    for idx, batch in enumerate(data_loader):
        x = batch['x']
        y = batch['y']
        f_name = batch['f_name']

        x = x.cuda()
        
        if task_type == 'classification':
            draw_cam(cam, x, y, n_class, writer)

        else:
            raise NotImplementedError

        if export_csv: # csv
            pred = model(x)
            writer.export_csv(f_name, y.cpu().item(), pred.argmax(1).cpu().item())

        pbar.update()

    writer.close()