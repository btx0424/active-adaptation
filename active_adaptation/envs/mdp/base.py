def reward(func):
    func.is_reward = True
    return func

def observation(func):
    func.is_observation = True
    return func

def termination(func):
    func.is_termination = True
    return func