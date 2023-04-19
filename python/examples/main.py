import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import ipopt

# Optimization function
def func():
    x = sym.MatrixSymbol('x', 2, 1)
    f_x = x.transpose()*sym.Matrix(np.eye(2))*x/2

    return f_x, x

def grad_func():
    f_x, x = func()

    return f_x.diff(x), x

def hessian_func():
    g_x, x = grad_func()

    return g_x.as_explicit().jacobian(x), x

def eval_func(x_eval):
    f_x, x = func()

    return f_x.subs({x: sym.Matrix(x_eval)}).doit()

def eval_grad_func(x_eval):
    g_x, x = grad_func()

    return np.array(g_x.subs({x: sym.Matrix(x_eval)})).astype(np.float64)

def eval_hessian_func(x_eval):
    h_x, x = hessian_func()

    return np.array(h_x.subs({x: sym.Matrix(x_eval)}).doit()).astype(np.float64)

# Constraints

#Equality constraint

#Inequality constraint
def ineq_func():
    __, x = func()
    c_x = sym.Matrix([[x[0]-1], [x[1]-1], [x[0]+x[1]-3]]).doit()

    return c_x, x

def grad_ineq_func():
    c_x, x = ineq_func()

    return c_x.jacobian(x), x

def eval_ineq_func(x_eval):
    c_x, x = ineq_func()

    return np.array(c_x.subs({x: sym.Matrix(x_eval)})).astype(np.float64)

def eval_grad_ineq_func(x_eval):
    A_x, x = grad_ineq_func()

    return np.array(A_x.subs({x: sym.Matrix(x_eval)})).astype(np.float64)

def main():
    x_eval = [[3], [3]]
    mu = 0.8
    
    lambda_ = mu/eval_ineq_func(x_eval)

    print("x_eval: \n" + str(x_eval))
    print("lambda_: \n" + str(lambda_))

    y_list = []
    N = 100
    for iter_ in range(0, N):
        f_x = eval_func(x_eval)[0]
        g_x = eval_grad_func(x_eval)
        W_x = eval_hessian_func(x_eval)

        c_x = eval_ineq_func(x_eval)
        A_x = eval_grad_ineq_func(x_eval)

        p = ipopt.solve(g_x, W_x, c_x, A_x, lambda_, mu)
        # p = ipopt.solve(g_x, W_x, c_x, None, None, mu)

        p_x = p[0:np.size(x_eval)]
        p_lambda = p[np.size(x_eval):]

        alpha = 1

        for l, p_l in zip(lambda_, p_lambda):
            if (p_l[0] < 0):
                alpha = np.max(np.min([-l[0]/p_l[0], alpha]), 0)

        x_eval = x_eval + alpha*p_x
        lambda_ = lambda_ + alpha*p_lambda

        y_list.append(f_x)

        print("f_x: \n" + str(f_x))
        print("p_x: \n" + str(p_x))
        print("p_lambda: \n" + str(p_lambda))
        print("x_eval: \n" + str(x_eval))
        print("lambda_: \n" + str(lambda_))
        print("alpha: "+str(alpha))
        print("mu: " + str(mu))
        mu = mu*0.8
    

    f_x = eval_func(x_eval)[0]
    g_x = eval_grad_func(x_eval)
    W_x = eval_hessian_func(x_eval)

    c_x = eval_ineq_func(x_eval)
    A_x = eval_grad_ineq_func(x_eval)

    y_list.append(eval_func(x_eval)[0])

    print("f_x: \n" + str(f_x))
    print("p_x: \n" + str(p_x))
    print("p_lambda: \n" + str(p_lambda))
    print("x_eval: \n" + str(x_eval))
    print("lambda_: \n" + str(lambda_))
    print("alpha: "+str(alpha))
    print("mu: " + str(mu))

    plt.plot(range(0, N+1), y_list)
    plt.show()

if __name__ == "__main__":
    main()