"""
Performance Statistics Module for Mamba4Rec
计算模型的性能统计指标：推理延迟、吞吐量、显存占用、参数量等
"""

import torch
import time
import numpy as np


def _move_batch_to_device(batch, device):
    """
    将RecBole的batch移动到指定设备
    RecBole的batch可能是Interaction对象或tuple，需要正确处理
    
    Args:
        batch: RecBole的batch（可能是Interaction对象或tuple）
        device: 目标设备（torch.device对象）
        
    Returns:
        处理后的batch和实际的interaction对象（都已移动到设备）
    """
    # 确保device是torch.device对象
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(batch, (tuple, list)):
        # tuple格式，提取interaction对象
        interaction = batch[0] if len(batch) > 0 else batch
    else:
        # 直接是Interaction对象
        interaction = batch
    
    # 确保interaction被移动到设备
    # RecBole的Interaction对象应该支持to()方法
    if hasattr(interaction, 'to'):
        try:
            interaction = interaction.to(device)
        except (AttributeError, TypeError, RuntimeError) as e:
            # 如果to()方法失败，尝试使用RecBole的默认方法
            # 或者手动移动tensor
            try:
                # 尝试通过字典方式访问（RecBole的Interaction可能支持）
                if hasattr(interaction, '__getitem__'):
                    # 手动移动所有tensor字段
                    moved = False
                    for key in list(interaction.keys()) if hasattr(interaction, 'keys') else []:
                        try:
                            value = interaction[key]
                            if isinstance(value, torch.Tensor) and value.device != device:
                                interaction[key] = value.to(device)
                                moved = True
                        except:
                            pass
                    if not moved:
                        # 如果无法移动，抛出错误以便调试
                        raise RuntimeError(f"Failed to move interaction to device {device}. Error: {e}")
            except:
                # 最后的尝试：直接使用batch.to(device)如果batch不是tuple
                if not isinstance(batch, (tuple, list)):
                    try:
                        interaction = batch.to(device)
                    except:
                        raise RuntimeError(f"Failed to move batch/interaction to device {device}. Original error: {e}")
    
    # 如果是tuple格式，更新batch
    if isinstance(batch, (tuple, list)):
        if len(batch) > 0:
            batch = (interaction,) + batch[1:] if len(batch) > 1 else (interaction,)
        else:
            batch = (interaction,)
    
    return batch, interaction


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        dict: 包含总参数量、可训练参数量等信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算参数量（MB）
    total_params_mb = total_params * 4 / (1024 ** 2)  # 假设float32，4字节
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_mb': total_params_mb,
        'total_params_m': total_params / 1e6  # 百万参数
    }


def get_memory_usage(device='cuda'):
    """
    获取显存占用情况
    
    Args:
        device: 设备类型 ('cuda' 或 'cpu')
        
    Returns:
        dict: 显存占用信息
    """
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
            'allocated_gb': allocated / 1024,
            'reserved_gb': reserved / 1024
        }
    else:
        # CPU模式下无法获取显存信息
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'max_allocated_mb': 0,
            'note': 'CPU mode, GPU memory not available'
        }


def measure_inference_latency(model, dataloader, device, num_warmup=10, num_runs=100):
    """
    测量推理延迟和吞吐量
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器（RecBole的dataloader）
        device: 设备
        num_warmup: 预热轮数（用于稳定GPU状态）
        num_runs: 实际测量轮数
        
    Returns:
        dict: 包含延迟和吞吐量统计信息
    """
    model.eval()
    
    # 获取ITEM_SEQ字段名
    ITEM_SEQ = model.ITEM_SEQ
    
    # 预热
    with torch.no_grad():
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= num_warmup:
                break
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            _ = model.predict(interaction)
            warmup_count += 1
    
    # 同步GPU（如果使用CUDA）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量推理时间
    latencies = []
    total_users = 0
    total_time = 0
    
    with torch.no_grad():
        start_time = time.time()
        run_count = 0
        for batch in dataloader:
            if run_count >= num_runs:
                break
            
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            batch_size = interaction[ITEM_SEQ].shape[0]
            
            # 单个batch的推理时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_start = time.time()
            
            _ = model.predict(interaction)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_end = time.time()
            
            batch_latency = (batch_end - batch_start) * 1000  # 转换为毫秒
            per_user_latency = batch_latency / batch_size  # 每个用户的延迟
            
            latencies.append(per_user_latency)
            total_users += batch_size
            run_count += 1
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
    
    if len(latencies) == 0:
        return {
            'avg_latency_ms': 0,
            'median_latency_ms': 0,
            'p95_latency_ms': 0,
            'p99_latency_ms': 0,
            'min_latency_ms': 0,
            'max_latency_ms': 0,
            'std_latency_ms': 0,
            'qps': 0,
            'total_users': 0,
            'total_time_sec': 0
        }
    
    latencies = np.array(latencies)
    
    # 计算统计信息
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 计算吞吐量 (QPS)
    qps = total_users / total_time if total_time > 0 else 0
    
    return {
        'avg_latency_ms': avg_latency,
        'median_latency_ms': median_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'std_latency_ms': np.std(latencies),
        'qps': qps,
        'total_users': total_users,
        'total_time_sec': total_time
    }


def measure_full_sort_latency(model, dataloader, device, num_warmup=5, num_runs=50):
    """
    测量全排序推理延迟（full_sort_predict，更耗时）
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        device: 设备
        num_warmup: 预热轮数
        num_runs: 实际测量轮数
        
    Returns:
        dict: 包含延迟和吞吐量统计信息
    """
    model.eval()
    ITEM_SEQ = model.ITEM_SEQ
    
    # 预热
    with torch.no_grad():
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= num_warmup:
                break
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            _ = model.full_sort_predict(interaction)
            warmup_count += 1
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    latencies = []
    total_users = 0
    total_time = 0
    
    with torch.no_grad():
        start_time = time.time()
        run_count = 0
        for batch in dataloader:
            if run_count >= num_runs:
                break
            
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            batch_size = interaction[ITEM_SEQ].shape[0]
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_start = time.time()
            
            _ = model.full_sort_predict(interaction)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_end = time.time()
            
            batch_latency = (batch_end - batch_start) * 1000
            per_user_latency = batch_latency / batch_size
            
            latencies.append(per_user_latency)
            total_users += batch_size
            run_count += 1
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
    
    if len(latencies) == 0:
        return {
            'avg_latency_ms': 0,
            'median_latency_ms': 0,
            'p95_latency_ms': 0,
            'qps': 0,
            'total_users': 0,
            'total_time_sec': 0
        }
    
    latencies = np.array(latencies)
    
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    qps = total_users / total_time if total_time > 0 else 0
    
    return {
        'avg_latency_ms': avg_latency,
        'median_latency_ms': median_latency,
        'p95_latency_ms': p95_latency,
        'qps': qps,
        'total_users': total_users,
        'total_time_sec': total_time
    }


def calculate_speedup_ratio(model, dataloader, device):
    """
    计算推理加速比
    由于没有上线系统，通过对比训练模式和推理模式的性能来评估加速效果
    推理模式通常比训练模式快，因为不需要计算梯度
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        dict: 训练模式vs推理模式的性能对比和加速比
    """
    ITEM_SEQ = model.ITEM_SEQ
    ITEM_SEQ_LEN = model.ITEM_SEQ_LEN
    num_samples = 20  # 测试样本数
    
    # 1. 训练模式性能（需要计算梯度）
    model.train()
    train_latencies = []
    
    with torch.set_grad_enabled(True):
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            batch_size = interaction[ITEM_SEQ].shape[0]
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            # 训练模式：需要计算loss
            seq_output = model.forward(interaction[ITEM_SEQ], interaction[ITEM_SEQ_LEN])
            # 模拟计算loss（简化）
            _ = seq_output.sum()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            latency = (end - start) * 1000 / batch_size
            train_latencies.append(latency)
            sample_count += 1
    
    # 2. 推理模式性能（不需要梯度）
    model.eval()
    inference_latencies = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            batch_size = interaction[ITEM_SEQ].shape[0]
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model.predict(interaction)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            latency = (end - start) * 1000 / batch_size
            inference_latencies.append(latency)
            sample_count += 1
    
    train_avg = np.mean(train_latencies) if train_latencies else 0
    inference_avg = np.mean(inference_latencies) if inference_latencies else 0
    
    # 计算加速比
    speedup_ratio = train_avg / inference_avg if inference_avg > 0 else 0
    
    return {
        'train_mode_latency_ms': train_avg,
        'inference_mode_latency_ms': inference_avg,
        'speedup_ratio': speedup_ratio,
        'note': 'Speedup ratio compares training mode (with gradients) vs inference mode (no gradients)'
    }


def get_quick_stats(model, dataloader, config, logger):
    """
    快速性能统计（轻量级版本，用于训练前检查）
    只计算关键指标，不进行完整的性能测试
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器（可以是valid_data或test_data）
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        dict: 快速统计信息
    """
    device = torch.device(config['device'])
    model.eval()
    
    stats = {}
    
    # 1. 模型参数量（快速）
    param_stats = count_parameters(model)
    stats['parameters'] = param_stats
    
    # 2. 显存占用（快速测试）
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    memory_after = get_memory_usage(device.type if device.type == 'cuda' else 'cpu')
    stats['memory'] = memory_after
    
    # 3. 快速推理延迟测试（只测试少量样本）
    ITEM_SEQ = model.ITEM_SEQ
    latencies = []
    test_samples = 0
    max_samples = 10  # 只测试10个batch
    
    with torch.no_grad():
        for batch in dataloader:
            if test_samples >= max_samples:
                break
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            batch_size = interaction[ITEM_SEQ].shape[0]
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model.predict(interaction)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            latency = (end - start) * 1000 / batch_size
            latencies.append(latency)
            test_samples += 1
    
    if latencies:
        latencies = np.array(latencies)
        stats['inference_latency'] = {
            'avg_latency_ms': np.mean(latencies),
            'samples_tested': test_samples,
            'note': 'Quick test with limited samples'
        }
    else:
        stats['inference_latency'] = {
            'avg_latency_ms': 0,
            'samples_tested': 0,
            'note': 'No samples tested'
        }
    
    return stats


def get_comprehensive_stats(model, test_data, config, logger):
    """
    获取模型的综合性能统计
    
    Args:
        model: 训练好的模型
        test_data: 测试数据集（RecBole的dataloader，已经是可迭代对象）
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        dict: 所有性能统计信息
    """
    device = torch.device(config['device'])
    model.eval()
    
    # test_data 已经是 dataloader，直接使用
    test_dataloader = test_data
    
    stats = {}
    
    # 1. 模型参数量
    logger.info("=" * 50)
    logger.info("Computing Model Parameters...")
    param_stats = count_parameters(model)
    stats['parameters'] = param_stats
    logger.info(f"Total Parameters: {param_stats['total_params']:,} ({param_stats['total_params_m']:.2f}M)")
    logger.info(f"Trainable Parameters: {param_stats['trainable_params']:,}")
    logger.info(f"Model Size (estimated): {param_stats['total_params_mb']:.2f} MB")
    
    # 2. 显存占用（推理前）
    logger.info("=" * 50)
    logger.info("Measuring GPU Memory Usage...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    memory_before = get_memory_usage(device.type if device.type == 'cuda' else 'cpu')
    
    # 进行一次推理以分配显存
    with torch.no_grad():
        for batch in test_dataloader:
            # 处理batch的设备移动
            batch, interaction = _move_batch_to_device(batch, device)
            _ = model.predict(interaction)
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    memory_after = get_memory_usage(device.type if device.type == 'cuda' else 'cpu')
    stats['memory'] = memory_after
    logger.info(f"GPU Memory Allocated: {memory_after['allocated_mb']:.2f} MB ({memory_after['allocated_gb']:.2f} GB)")
    logger.info(f"GPU Memory Reserved: {memory_after['reserved_mb']:.2f} MB ({memory_after['reserved_gb']:.2f} GB)")
    if 'max_allocated_mb' in memory_after and memory_after['max_allocated_mb'] > 0:
        logger.info(f"Peak GPU Memory: {memory_after['max_allocated_mb']:.2f} MB")
    if 'note' in memory_after:
        logger.info(f"Note: {memory_after['note']}")
    
    # 3. 推理延迟和吞吐量（单样本预测）
    logger.info("=" * 50)
    logger.info("Measuring Inference Latency (per user) and Throughput...")
    latency_stats = measure_inference_latency(model, test_dataloader, device, num_warmup=10, num_runs=100)
    stats['inference_latency'] = latency_stats
    logger.info(f"Average Latency per User: {latency_stats['avg_latency_ms']:.3f} ms")
    logger.info(f"Median Latency per User: {latency_stats['median_latency_ms']:.3f} ms")
    logger.info(f"P95 Latency per User: {latency_stats['p95_latency_ms']:.3f} ms")
    logger.info(f"P99 Latency per User: {latency_stats['p99_latency_ms']:.3f} ms")
    logger.info(f"Throughput (QPS): {latency_stats['qps']:.2f} queries/sec")
    
    # 4. 全排序推理延迟（如果需要）
    logger.info("=" * 50)
    logger.info("Measuring Full Sort Inference Latency...")
    full_sort_stats = measure_full_sort_latency(model, test_dataloader, device, num_warmup=5, num_runs=30)
    stats['full_sort_latency'] = full_sort_stats
    logger.info(f"Full Sort - Average Latency: {full_sort_stats['avg_latency_ms']:.3f} ms")
    logger.info(f"Full Sort - Throughput: {full_sort_stats['qps']:.2f} queries/sec")
    
    # 5. 推理加速比分析（训练模式 vs 推理模式）
    logger.info("=" * 50)
    logger.info("Analyzing Speedup Ratio (Training Mode vs Inference Mode)...")
    logger.info("Note: Since the system is not deployed, we compare training mode (with gradients)")
    logger.info("      vs inference mode (no gradients) as a proxy for speedup analysis.")
    logger.info("      Inference mode is typically faster due to no gradient computation.")
    speedup_stats = calculate_speedup_ratio(model, test_dataloader, device)
    stats['speedup_analysis'] = speedup_stats
    logger.info(f"Training Mode Latency: {speedup_stats['train_mode_latency_ms']:.3f} ms/user")
    logger.info(f"Inference Mode Latency: {speedup_stats['inference_mode_latency_ms']:.3f} ms/user")
    logger.info(f"Speedup Ratio (Train/Inference): {speedup_stats['speedup_ratio']:.2f}x")
    
    logger.info("=" * 50)
    
    return stats

# import time
# import torch
# import numpy as np
# from recbole.utils import set_color

# def get_quick_stats(model, valid_data, config, logger):
#     """快速性能检查"""
#     stats = {
#         'parameters': {},
#         'memory': {},
#         'inference_latency': {}
#     }
    
#     # 参数统计
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     stats['parameters']['total_params'] = total_params
#     stats['parameters']['total_params_m'] = total_params / 1e6
    
#     # 内存使用
#     if torch.cuda.is_available():
#         torch.cuda.reset_peak_memory_stats()
#         allocated = torch.cuda.memory_allocated() / 1024**3
#         stats['memory']['allocated_gb'] = allocated
#     else:
#         stats['memory']['note'] = "CUDA not available"
    
#     # 推理延迟测试
#     try:
#         model.eval()
#         test_batch = next(iter(valid_data))
        
#         start_time = time.time()
#         with torch.no_grad():
#             _ = model.predict(test_batch)
#         end_time = time.time()
        
#         latency_ms = (end_time - start_time) * 1000
#         stats['inference_latency']['avg_latency_ms'] = latency_ms
#         stats['inference_latency']['samples_tested'] = 1
        
#     except Exception as e:
#         logger.warning(f"Inference test failed: {e}")
#         stats['inference_latency']['error'] = str(e)
    
#     return stats

# def get_comprehensive_stats(model, test_data, config, logger):
#     """综合性能统计"""
#     stats = {
#         'parameters': {},
#         'memory': {},
#         'inference_latency': {},
#         'full_sort_latency': {},
#         'speedup_analysis': {}
#     }
    
#     device = config['device']
#     batch_size = config['eval_batch_size']
    
#     # 1. 参数统计
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     stats['parameters']['total_params'] = total_params
#     stats['parameters']['trainable_params'] = trainable_params
#     stats['parameters']['total_params_m'] = total_params / 1e6
#     stats['parameters']['trainable_params_m'] = trainable_params / 1e6
    
#     # 2. 内存使用
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
        
#         # 测试一个batch的内存使用
#         try:
#             test_batch = next(iter(test_data))
#             with torch.no_grad():
#                 _ = model.predict(test_batch)
            
#             allocated = torch.cuda.memory_allocated() / 1024**3
#             cached = torch.cuda.memory_reserved() / 1024**3
#             peak = torch.cuda.max_memory_allocated() / 1024**3
            
#             stats['memory']['allocated_gb'] = allocated
#             stats['memory']['cached_gb'] = cached
#             stats['memory']['peak_gb'] = peak
            
#         except Exception as e:
#             logger.warning(f"Memory test failed: {e}")
#             stats['memory']['error'] = str(e)
#     else:
#         stats['memory']['note'] = "CUDA not available"
    
#     # 3. 推理延迟测试
#     model.eval()
#     latencies = []
    
#     try:
#         test_loader = test_data.get_loader(batch_size=batch_size, shuffle=False)
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(test_loader):
#                 if batch_idx >= 10:  # 测试10个batch
#                     break
                
#                 batch = {k: v.to(device) if torch.is_tensor(v) else v 
#                         for k, v in batch.items()}
                
#                 start_time = time.perf_counter()
#                 _ = model.predict(batch)
#                 torch.cuda.synchronize() if torch.cuda.is_available() else None
#                 end_time = time.perf_counter()
                
#                 latency_ms = (end_time - start_time) * 1000
#                 latencies.append(latency_ms)
        
#         if latencies:
#             stats['inference_latency']['avg_latency_ms'] = np.mean(latencies)
#             stats['inference_latency']['std_latency_ms'] = np.std(latencies)
#             stats['inference_latency']['min_latency_ms'] = np.min(latencies)
#             stats['inference_latency']['max_latency_ms'] = np.max(latencies)
#             stats['inference_latency']['qps'] = 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
#             stats['inference_latency']['samples_tested'] = len(latencies)
            
#     except Exception as e:
#         logger.warning(f"Latency test failed: {e}")
#         stats['inference_latency']['error'] = str(e)
    
#     # 4. Full-sort延迟测试
#     try:
#         full_sort_latencies = []
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(test_loader):
#                 if batch_idx >= 5:  # 测试5个batch
#                     break
                
#                 batch = {k: v.to(device) if torch.is_tensor(v) else v 
#                         for k, v in batch.items()}
                
#                 start_time = time.perf_counter()
#                 _ = model.full_sort_predict(batch)
#                 torch.cuda.synchronize() if torch.cuda.is_available() else None
#                 end_time = time.perf_counter()
                
#                 latency_ms = (end_time - start_time) * 1000
#                 full_sort_latencies.append(latency_ms)
        
#         if full_sort_latencies:
#             stats['full_sort_latency']['avg_latency_ms'] = np.mean(full_sort_latencies)
#             stats['full_sort_latency']['std_latency_ms'] = np.std(full_sort_latencies)
#             stats['full_sort_latency']['samples_tested'] = len(full_sort_latencies)
            
#     except Exception as e:
#         logger.warning(f"Full-sort latency test failed: {e}")
    
#     # 5. 加速比分析（如果可用）
#     try:
#         # 简单估计：推理时间 vs 理论训练时间
#         if 'avg_latency_ms' in stats['inference_latency']:
#             # 假设训练时间大约是推理时间的10倍（由于梯度计算）
#             train_time_est = stats['inference_latency']['avg_latency_ms'] * 10
#             stats['speedup_analysis']['estimated_train_ms_per_batch'] = train_time_est
#             stats['speedup_analysis']['speedup_ratio'] = 10.0  # 保守估计
            
#     except Exception as e:
#         pass
    
#     return stats

# def print_performance_summary(stats, logger):
#     """打印性能摘要"""
#     logger.info("\n" + "=" * 60)
#     logger.info(set_color("PERFORMANCE SUMMARY", "green", bold=True))
#     logger.info("=" * 60)
    
#     # 参数
#     if 'total_params_m' in stats['parameters']:
#         logger.info(f"📊 Model Size: {stats['parameters']['total_params_m']:.2f}M parameters")
    
#     # 内存
#     if 'allocated_gb' in stats['memory']:
#         logger.info(f"💾 GPU Memory: {stats['memory']['allocated_gb']:.2f} GB (peak: {stats['memory'].get('peak_gb', 0):.2f} GB)")
    
#     # 延迟
#     if 'avg_latency_ms' in stats['inference_latency']:
#         logger.info(f"⚡ Inference: {stats['inference_latency']['avg_latency_ms']:.2f} ms/user (±{stats['inference_latency'].get('std_latency_ms', 0):.2f})")
#         logger.info(f"📈 Throughput: {stats['inference_latency'].get('qps', 0):.2f} users/sec")
    
#     if 'avg_latency_ms' in stats.get('full_sort_latency', {}):
#         logger.info(f"🔍 Full-sort: {stats['full_sort_latency']['avg_latency_ms']:.2f} ms/user")
    
#     logger.info("=" * 60)
