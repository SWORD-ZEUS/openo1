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

def process_state_dict4verifier(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.response'):
            new_key = 'model.lm_head' + key[len('model.response'):]
        else:
            new_key = key
        if key.startswith("model.win_rate") or key.startswith("model.classification"):
            new_key = key[len("model."):]
        new_state_dict[new_key] = value
    return new_state_dict