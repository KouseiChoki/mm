# VFIMamba 多机单卡 DDP

`train_ddp.py` 是独立训练入口，不修改原有 `train.py` 的使用方式。

## 1. 环境要求

所有机器需要：

- 相同的代码 commit、Python/PyTorch/CUDA 和依赖版本。
- 一张可用 CUDA GPU。
- 训练数据与清单内容一致。数据可各自放在本地 SSD，不需共享文件系统。
- 配置文件中 `root` / `lists_dir` 在每台机器上均有效。
- 节点可访问主节点的 rendezvous 端口，例如 `29500`。

可先在其他节点测试：

```bash
nc -vz 192.168.1.10 29500
```

## 2. 三台机器启动示例

假设主节点 IP 为 `192.168.1.10`，三台机器的 `node_rank` 分别为
`0` / `1` / `2`。

主节点：

```bash
torchrun \
  --nnodes=3 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train_ddp.py --config train_config_ddp.yaml \
  --restore_ckpt /path/to/model.pkl
```

第2台：

```bash
torchrun \
  --nnodes=3 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train_ddp.py --config train_config_ddp.yaml \
  --restore_ckpt /path/to/model.pkl
```

第3台：

```bash
torchrun \
  --nnodes=3 \
  --nproc_per_node=1 \
  --node_rank=2 \
  --master_addr=192.168.1.10 \
  --master_port=29500 \
  train_ddp.py --config train_config_ddp.yaml \
  --restore_ckpt /path/to/model.pkl
```

从头训练时删除 `--restore_ckpt`，并将 YAML 中的 `optim.finetune` 改为
`false`。

首次部署建议先在每台机器分别做单进程冒烟测试：

```bash
torchrun --standalone --nproc_per_node=1 \
  train_ddp.py --config train_config_ddp.yaml \
  --restore_ckpt /path/to/model.pkl
```

确认数据、BF16 forward/backward 和 checkpoint 路径都正常后，再启动多机。

## 3. 精确恢复 DDP checkpoint

DDP 入口保存的 checkpoint 同时包含网络、optimizer、scaler、epoch、step、
各 rank 的 RNG 状态和当前 best 指标。
中断恢复时在所有节点使用：

```bash
train_ddp.py \
  --config train_config_ddp.yaml \
  --restore_ckpt /path/to/ddp_checkpoint.pkl \
  --resume
```

普通 finetune 不要传 `--resume`，否则会连 optimizer/epoch/step 一起恢复。
要保持 LR schedule 严格连续，resume 时应保持原来的 `world_size`、
`grad_accum_steps` 和 `preserve_single_node_samples_per_epoch` 设置。

## 4. 网卡选择和调试

如果机器同时有多个网卡，每台机器分别设置实际的有线网卡：

```bash
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO
```

通过以下命令查看网卡名：

```bash
ip addr
```

稳定后可取消 `NCCL_DEBUG=INFO`。

## 5. 全局 batch 与局域网同步

```text
global batch = YAML每卡batch * world_size * grad_accum_steps
```

`grad_accum_steps=1` 表示每个micro-batch都同步。如果希望减少局域网梯度
同步频率，可设为 `2` 或 `4`。代码会在前 N-1 个micro-batch上使用
`DDP.no_sync()`，只在最后一次 backward 同步。

`preserve_single_node_samples_per_epoch=true` 会将 optimizer steps/epoch 按
`world_size * grad_accum_steps` 缩减，使每个epoch处理的总样本量接近单机设置。

如果仍然需要降低通信量，可将 `distributed.fp16_compress_hook` 设为
`true`，以 FP16 通信梯度bucket后再还原。该选项会引入轻微压缩误差，
建议先用默认 `false` 跑基线。

## 6. 运行规则

- crop 只由 rank0 抽取，并广播给所有节点。
- 每个rank使用不同随机种子和 DataLoader worker 种子。
- 验证由所有rank分片执行，PSNR汇总到rank0。
- `data.val_lists` 可配置 `easy/normal/hard/opensource/teacher` 等独立验证清单；
  缺失的可选清单会跳过，teacher 清单还会统计总/运动区/静态区 flow EPE。
- `monitor.best_metric` 决定 best checkpoint；`best_mode: auto` 对 PSNR 取最大，
  对 EPE/loss 取最小。
- TensorBoard、异常dump、checkpoint和主训练日志只由rank0写入。
- 任何一个节点退出都会使整个DDP作业停止。
