def _save_state(param_dict):
    rtr = []
    for tran, param_set in param_dict.items():
        for param_name in param_set:
            rtr.append((tran, param_name, tran.get_parameter(param_name)))
    return rtr


def _restore_state(param_state):
    for (tran, param_name, param_value) in param_state:
        tran.set_parameter(param_name, param_value)