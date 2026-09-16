import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
from tqdm import tqdm

from metrics import all_metrics


# 构建 DataLoader
def build_dataloaders(train_dataset, dev_dataset, test_dataset, tokenizer, batch_size: int, seed: int):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        generator=g,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_loader, dev_loader, test_loader


# 构建 AdamW 优化器
def build_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": config["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=config["learning_rate"])


# 构建带 warmup 的线性衰减调度器
def build_scheduler(optimizer, config, num_training_steps):
    warmup_steps = int(num_training_steps * config["warmup_ratio"])
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


# 单个 epoch 训练
def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch: int):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# 完整训练流程
def train(model, train_loader, dev_loader, optimizer, scheduler, device, config: dict, id2label: dict, save_dir: str, log_fn=None):
    best_macro_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{config['num_epochs']} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        eval_preds, eval_labels = predict(model, dev_loader, device)
        eval_metrics = all_metrics(eval_labels, eval_preds, id2label, prefix="eval_")

        print(f"eval_accuracy    = {eval_metrics['eval_accuracy']:.4f}")
        print(f"eval_macro_f1    = {eval_metrics['eval_macro_f1']:.4f}")
        print(f"eval_weighted_f1 = {eval_metrics['eval_weighted_f1']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **eval_metrics,
        })

        if log_fn is not None:
            log_fn({"train_loss": train_loss, **eval_metrics})

        # 根据验证集 macro_f1 保存最佳模型
        if eval_metrics["eval_macro_f1"] > best_macro_f1:
            best_macro_f1 = eval_metrics["eval_macro_f1"]
            best_epoch = epoch

            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"保存新的最佳模型到 {save_dir}")

    print(f"\n训练结束。\n 最佳 epoch : {best_epoch}, 最佳验证集: macro_f1 = {best_macro_f1:.4f}")
    return best_epoch, best_macro_f1, history


# 测试集预测
def predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="Predict", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)
