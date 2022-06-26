import torch
class reverse_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return  - (grad_output.clamp(-10, 10))
rg = reverse_grad.apply
def grad_reverse(x):
    return rg(x)


'''
class grad_reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        print("BB")
        return x.clone()
    
    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads * 10
        return dx, None
'''
'''
class GradReverse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        #print("FORWARD!")
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        grad_input = grad_input.clamp(-0.5, 0.5)
        #print(grad_input.max().item(), grad_input.min().item())
        #print(grad_input)
        return  - lambda_ * grad_input, None
    
def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x,lam)

'''