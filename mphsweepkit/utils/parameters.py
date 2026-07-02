import mph


def get_parameters(model: mph.Model) -> dict:
    """
    Get all global parameters and their values.

    :param model: The COMSOL model object.
    :return: A dictionary containing parameter names and their corresponding values."""
    parameters = {}
    for name in model.parameters().keys():
        value = model.parameters().get(name)
        parameters[name] = value
    return parameters


def print_parameters(parameters: dict):
    """
    Print the parameters in a readable format.

    :param parameters: A dictionary containing parameter names and their corresponding values."""
    for name, value in parameters.items():
        print(f"{name}: {value}")


def get_descriptions(model: mph.Model, parameter_names: list) -> dict:
    """
    Get the COMSOL descriptions for a list of parameter names.

    :param model: COMSOL model used to resolve the parameter descriptions.
    :param parameter_names: List of parameter names to describe.
    :return: A dictionary mapping parameter names to their descriptions.
    """
    return {name: model.description(name) for name in parameter_names}
