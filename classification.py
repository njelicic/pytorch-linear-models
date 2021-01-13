class LogisticRegression():
    def __init__(self, C=0.1,lr=1e-3, penalty=None, n_iter=10000):
        self.C = C
        self.lr = lr
        self.history = []
        self.penalty = penalty 
        self.n_iter = n_iter
        return
    
    def logreg(self, x):
    
        """"Logistic regression function"""  
         
        return 1/(1+ torch.exp(-(x @ self.w.t() + self.b)))            
    
    def loss(self, y, y_hat):
        """"Calculate loss"""  
        
        bce = -torch.mean(y * torch.log(y_hat))              #compute binary crossentropy
        
        if self.penalty == 'l2':
            penalty = self.C*torch.sum(self.w**2)            # lambda multiplied by the sum of squared weights 
        
        if self.penalty == 'l1':
            penalty = torch.abs(self.C*torch.sum(self.w))    # lambda multiplied by the sum of weights 
        
        if self.penalty == None:
            penalty = 0 
        
        return  bce + penalty 
    
    
    def cast_to_tensor(self, x):
        return torch.tensor(x).float()
    
    def fit(self,x,y):
        """"Fit model"""  
        x = self.cast_to_tensor(x)
        y = self.cast_to_tensor(y)
        
        self.w = torch.randn(x.size()[1], requires_grad=True) #instantiate weights
        self.b = torch.randn(1, requires_grad=True)           #instantiate bias
        
        for i in range(self.n_iter):
            y_hat = self.logreg(x)    # make predictions
            loss = self.loss(y,y_hat) # calculate loss function
            loss.backward()           # backprop
            
            with torch.no_grad(): 
                self.w -= self.w.grad * self.lr #update weights
                self.b -= self.b.grad * self.lr #update bias
                self.w.grad.zero_()
                self.b.grad.zero_()
            
            self.history.append(loss.item())
            
    def predict(self, x):
        """"Predict"""  
        x = self.cast_to_tensor(x)
        return self.logreg(x).detach().numpy()
    

    
    def plot_history(self):
        """"Plot loss function over time"""  
        return sns.lineplot(x=[i+1 for i in range(len(self.history))],y=self.history).set(xlabel='Iteration', ylabel='Loss',title='History')
