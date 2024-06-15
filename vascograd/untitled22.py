from vascograd.nn import MLP
from vascograd.engine import Value

class Data:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

class Optimizer:
    
    def __init__(self, model: MLP, learning_rate: float = 1e-3) -> None:
        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        for p in self.model.parameters():
            p.data -= self.learning_rate * p.grad

    def zero_grad(self):
        for p in self.model.parameters():
            p.grad -= 0.0


class Learner:
    """Model + Optimizer"""
    #def __init__(self, model: MLP, optimizer: Optimizer):
    def __init__(self, model: MLP):
        self.model = model
        #self.optimizer = optimizer(model)
        self.optimizer = Optimizer(self.model)

    def predict(self, x):
        return self.model(x)

    def update(self, loss: Value) -> None:
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Evaluator:
    """Loss"""
    def __init__(self, loss_function):
        self.loss_fn = loss_function

    def get_loss(self, y, y_hat):
        return self.loss_fn(y_hat, y)

class Trainer:
   
    def __init__(self, data: Data, learner: Learner, evaluator: Evaluator):
        self.data = data
        self.learner = learner
        self.evaluator = evaluator

    def one_epoch(self):
        y_hat = [self.learner.predict(x) for x in self.data.X]
        loss = self.evaluator.get_loss(self.data.y, y_hat)
        self.learner.update(loss)
        return loss
    
    def run(self, n_epochs : int):
        for epoch in range(n_epochs):
            loss = self.one_epoch()
            if epoch % 10 == 0:
                print(f' epoch: {epoch} | loss : {loss.data}')
        print("Done!")