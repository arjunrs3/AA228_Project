import numpy as np    
import matplotlib.pyplot as plt
import pickle

class LinearAdvection: 
    # Initialize LA matrix 
    A = 0

    def __init__(self, c, max_CFL, T):
        self.c = c 
        self.max_CFL = max_CFL
        self.T = T 
    
    def deltaT(self, delta_x):
        """
        gets the deltaT for a given max CFL  
        """

        delta_T =  self.max_CFL * min(delta_x[1:-1]) / self.c
        CFL = self.c * delta_T / delta_x 
        return delta_T, CFL

    def linear_advection_step(self, u0, x, end_time):
        """
        Performs a step of the linear advection equations using a first order upwind 
            stencil 
        """

        flag = False
        
        if self.c > 0: 
            delta_x = x - np.roll(x, 1)
        else: 
            delta_x = np.roll(x, -1) - x

        deltaT, CFL = self.deltaT(delta_x)

        if end_time == -1: 
            flag = True 

        elif self.T + deltaT > end_time:
            flag = True
            deltaT = end_time - self.T 
            
            CFL = self.c * deltaT / delta_x

        # use upwinding 
        if self.c > 0: 
            u1 = u0 - CFL * (u0 - np.roll(u0, 1))
        else: 
            u1 = u0 - CFL * (np.roll(u0, -1) - u0)

        # Implement periodic boundary condition 
        u1[-1] = u1[-2]
        u1[0] = u1[1]

        # Update time 
        self.T = self.T + deltaT
        return u1, flag
    
    def solve_linear_advection(self, u0, x, end_time = 0.2, plotting = False): 
        """
        Carries out steps until the end_time is reached 
        If end_time = -1, only one step will be evaluated
        """
        if plotting: 
            self.plot(u0, x)
        u1 = u0
        while True:   
            u1, flag = self.linear_advection_step(u1, x, end_time)
            if plotting: 
                self.plot(u1, x)
            if flag: 
                break
        return u1 
    
    def plot(self, u, x): 
        """
        Plot the intermediate solutions to the advection equation 
            (i.e., after every linear_advection_step)
        """
        x_ex = np.linspace(0, 2, 100)
        u_ex = 1-np.tanh(2*(1-4*(x_ex-1/4 - self.c * self.T)))
        plt.figure(1)
        plt.cla()
        plt.plot(x_ex, u_ex, color = 'black')
        plt.plot(x, u, "o-")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Time: " + str(self.T)) 
        plt.pause(0.001)

if __name__ == "__main__": 
    """
    Simple test case of the linear advection code 
    """

    x = np.linspace(0, 1, 20)
    with open("new_x", "rb") as f: 
        x = pickle.load(f)

    x = np.linspace(0, 2, 20)
    u0 = 1-np.tanh(2*(1-4*(x-1/4)))
    x_ex = np.linspace(0, 2, 100)
    u_ex = 1-np.tanh(2*(1-4*(x_ex-1/4 - 0.5)))

    LA = LinearAdvection(1, 0.8, 0)
    u1 = LA.solve_linear_advection(u0, x, end_time = 0.5, plotting=True) 

    plt.figure(1)
    plt.cla()
    plt.plot(x_ex, u_ex, color = 'black')
    plt.plot(x, u1, "o-")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()




