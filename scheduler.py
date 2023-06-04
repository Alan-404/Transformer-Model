import torch.optim as optim

class Scheduler:
    def __init__(self, optimizer: optim.Optimizer, d_model: int, wramup_steps: int) -> None:
        self._optimizer = optimizer
        self.d_model = d_model
        self.wramup_steps = wramup_steps
        self.n_steps = 0

    def step(self):
        self.update()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_learning_rate(self):
        return (self.d_model ** (-0.5)) * min(self.n_steps ** (-0.5), self.n_steps * self.wramup_steps ** (-1.5))
    
    def update(self):
        self.n_steps += 1
        lr = self.get_learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
