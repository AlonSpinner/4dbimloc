import yaml
import numpy as np

def load_parameters(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            parameters_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return (exc)
    return parameters_dict

def get_actions(parameters):
    '''
    eval is evil. we use variables implictly
    '''
    DT = parameters['DT']
    velocity_factor = parameters['velocity_factor']
    straight = [eval(parameters['straight'])]
    turn_left = [eval(parameters['turn_left'])]
    turn_right = [eval(parameters['turn_right'])]
    actions  : str = parameters['actions']
    return eval(actions)

def get_U_COV(parameters):
    '''
    eval is evil. we use variables implictly
    '''
    velocity_factor = parameters['velocity_factor']
    U_COV = eval(parameters['U_COV'])
    return U_COV