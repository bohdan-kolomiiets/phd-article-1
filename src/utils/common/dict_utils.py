
def get_nested(value: dict, key_path: str, default: any):
    keys = key_path.split(":")
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    return value