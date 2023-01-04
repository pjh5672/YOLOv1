import torch


def resume_state(ckpt_path, rank, model, ema, optimizer, scheduler, scaler):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    start_epoch = ckpt["running_epoch"] + 1
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(rank)
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema and ckpt.get("ema_state"):
        ema.updates = ckpt['ema_update']
        ema.module.load_state_dict(ckpt['ema_state'], strict=True)
    return start_epoch