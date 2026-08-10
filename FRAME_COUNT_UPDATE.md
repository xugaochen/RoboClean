# 帧数统计显示更新说明

## 修改内容

### 1. 首页统计

**修改前**: 使用metadata中的`length`字段统计帧数
```python
episode_lengths[i] = ep.get("length", 0)  # 原视频帧数
```

**修改后**: 使用parquet中的实际行数统计帧数
```python
episode_table = loader.load_episode(i)
episode_lengths[i] = episode_table.num_rows  # 清洗后的帧数
```

**效果**:
- 总帧数显示清洗后的实际帧数
- 每个episode显示清洗后的帧数

### 2. Episode详情页

**新增显示**:
```
Episode 索引: 0
清洗后帧数: 333 帧
原视频帧数: 349 帧 (已清洗 16 帧)
任务: pick the bread
```

**逻辑**:
- 如果清洗后帧数 ≠ 原视频帧数 → 显示绿色提示"(已清洗 X 帧)"
- 如果清洗后帧数 = 原视频帧数 → 显示灰色提示"(未清洗)"

## 数据来源

- **清洗后帧数**: `load_episode(episode_idx).num_rows` - parquet文件中的实际行数
- **原视频帧数**: `episodes[episode_idx]['length']` - metadata中记录的原始帧数

## 优势

1. **准确统计** - 统计数据反映实际可用的帧数
2. **清晰展示** - 用户能清楚看到清洗效果
3. **信息透明** - 同时显示原视频帧数，不丢失信息

## 示例

### 未清洗的Episode
```
清洗后帧数: 349 帧
原视频帧数: 349 帧 (未清洗)
```

### 已清洗的Episode
```
清洗后帧数: 333 帧
原视频帧数: 349 帧 (已清洗 16 帧)
```

---

修改时间: 2026-08-10
影响文件: `roboclean/web/app.py`, `roboclean/web/templates/episode.html`