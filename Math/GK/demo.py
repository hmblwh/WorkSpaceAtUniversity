import numpy as np

def cost(x):
    x = np.array(x,dtype=np.float64)
    return (np.exp(x)-2/np.exp(x))**2

def grad(x):
    return 2*(np.exp(2*x)-4/np.exp(2*x))

def Gradient_Descent(alpha,x0,gra = 1e-3 ,loop=1000):
    x = x0
    for i in range(loop):
        x_new = x - alpha*grad(x)
        if abs(grad(x_new)) < gra:
            break
        x = x_new
    return x

print(Gradient_Descent(0.01,1.5))

