def load_yaml(file_path: str) -> dict:
    import yaml

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
