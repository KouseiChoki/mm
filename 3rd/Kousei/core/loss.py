import torch
MAX_FLOW = 400
import os
def sequence_loss_twoframes(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    
    
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """
    #print(flow_gt.shape, valid.shape, flow_preds[0].shape)
    #exit()

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    B, N, _, H, W = flow_gt.shape

    NAN_flag = False
    enhance_mask= None
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=2).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        flow_pre = flow_preds[i]
        i_loss = (flow_pre - flow_gt).abs()

        # if torch.isnan(i_loss).any():
        #     NAN_flag = True

        _valid = valid[:, :, None]
        if cfg.filter_epe:
            loss_mag = torch.sum(i_loss**2, dim=2).sqrt()
            mask = loss_mag > 1000
            #print(mask.shape, _valid.shape)
            if torch.any(mask):
                print("[Found extrem epe. Filtered out. Max is {}. Ratio is {}]".format(torch.max(loss_mag), torch.mean(mask.float())))
                _valid = _valid & (~mask[:, :, None])
        
        # if cfg.enhance_epe:
        #     mask = torch.ones_like(flow_gt)*255
        #     gradient_magnitude = torch.sqrt(flow_gt[:,:,0,:,:]**2 + flow_gt[:,:,1,:,:]**2)
        #     gradient_magnitude_x,gradient_magnitude_y = flow_gt[:,:,0,:,:],flow_gt[:,:,1,:,:]
        #     tt_x,tt_y = torch.gradient(gradient_magnitude_x, axis=(-1, -2))
        #     yy_x,yy_y = torch.gradient(gradient_magnitude_y, axis=(-1, -2))
        #     uu_x,uu_y = torch.gradient(gradient_magnitude, axis=(-1, -2))
        #     flow_x,flow_y = flow_pre[:,:,0,:,:],flow_pre[:,:,1,:,:]
        #     threshold = 1
        #     transition_region = ((tt_x > threshold) | (tt_y > threshold) | (yy_x > threshold) | (yy_y > threshold) | (uu_x>threshold) | (uu_y>threshold)) & ((flow_x.abs() < 10) | (flow_y.abs() < 10))
        #     transition_region = torch.cat([transition_region.unsqueeze(2)] * 2, dim=2)
        #     i_loss[transition_region] *= 1000
        #     mask[transition_region] = 0
        #     if enhance_mask is None:
        #         enhance_mask = mask[:,:,0,:,:].cpu().numpy()

        flow_loss += i_weight * (_valid * i_loss).mean()
    
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=2).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, NAN_flag

# def smooth_l1_loss(diff):
#     cond = diff.abs() < 1
#     loss = torch.where(cond, 0.5*diff**2, diff.abs()-0.5)
#     return loss

# def sequence_loss_smooth(flow_preds, flow_gt, valid, cfg):
#     """ Loss function defined over sequence of flow predictions """

#     gamma = cfg.gamma
#     max_flow = cfg.max_flow
#     n_predictions = len(flow_preds)    
#     flow_loss = 0.0
#     flow_gt_thresholds = [5, 10, 20]

#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#     valid = (valid >= 0.5) & (mag < max_flow)

#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         i_loss = smooth_l1_loss((flow_preds[i] - flow_gt))
#         flow_loss += i_weight * (valid[:, None] * i_loss).mean()

#     epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]

#     metrics = {
#         'epe': epe.mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#     }

#     flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
#     flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
#     for t in flow_gt_thresholds:
#         e = epe[flow_gt_length < t]
#         metrics.update({
#                 f"{t}-th-5px": (e < 5).float().mean().item()
#         })


#     return flow_loss, metrics

