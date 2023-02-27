import yaml
def load_parameters(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            parameters_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return (exc)
    return parameters_dict