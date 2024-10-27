def process_state_dict(state_dict):
    """处理状态字典的键，移除多余的 'model.model.' 前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        # 移除 'model.' 前缀
        if key.startswith('model.'):
            new_key = key[len('model.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict