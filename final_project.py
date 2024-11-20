from linear_advection import LinearAdvection
from scipy.integrate import simpson
from neural_net import NeuralNet
import numpy as np 
import tqdm
import pickle
import pandas as pd 
import matplotlib.pyplot as plt 
import copy 

class RL_agent: 
    """
    RL_agent that learns a relationship between observed quantities and action rewards 
    Acts on a mesh and decides whether to refine or not 
    """
    def __init__(self, 
                 gamma_c, 
                 Q_r,
                 Q_c,
                 max_elements=40,
                 problems = [LinearAdvection(1, 0.8, 0)],
                 endtimes = [0.5],
                 u0s = [lambda x: 1-np.tanh(2*(1-4*(x-1/4)))]):
        """
        Parameters: 
        -----------
        gamma_c: float 
            hyperparameter which controls computational cost to solution accuracy tradeoff 

        Q_r: NeuralNet
            The Q function neural net for refinement (trained or untrained)

        Q_c: NeuralNet 
            The Q function neural net for coarsening (trained or untrained)

        max_elements: int
            The maximum number of points after refinement 
            Defaults to 40

        problems: List(Objects): 
            A list of problems that can be used for training and evaluation 
            Defaults to a single simple linear advection problem 

        endtimes: List(times)
            A list of endtimes for each project
            -1 represents running 1 timestep 
            Defaults to [0.2] 

        u0s: List(functions)
            A list of functions that transforms an initial array of 1D coordinates 
                into an initial condition
            Defaults to a hyperbolic tangent centered at 0.5
        """
        self.gamma_c = gamma_c 
        self.max_elements = max_elements
        self.training_X = [] 
        self.training_y = [] 
        self.problems = problems
        self.endtimes = endtimes
        self.u0s = u0s
        self.Q_r = Q_r
        self.Q_c = Q_c

    def gen_training_data(self, x0, n_iterations):
        """
        Generates pairs of solution inputs and reward outputs for training the neural network 
        Calls train_step for the required number of iterations

        Parameters: 
        -----------
        x0: np.ndarray
            The initial (unrefined) array of x coordinates 

        n_iterations: int 
            The number of data points to generate 

        Returns: 
        -------- 
        training_refine_inputs: np.ndarray 
            The defined inputs to the neural network for refinement 
                (specified in train_step function)

        training_refine_outputs: np.ndarray 
            The reward for refinement (nn+r output)

        training_coarsen_inputs: np.ndarray 
            The defined inputs to the neural network for coarsening 

        training_coarsen_outputs: np.ndarray 
            The reward for coarsening (nn_c output)
        """
        training_refine_inputs = np.empty((n_iterations, 8))
        training_refine_outputs = np.empty(n_iterations)
        training_coarsen_inputs = np.empty((n_iterations * 2 - 2, 8))
        training_coarsen_outputs = np.empty(n_iterations * 2 - 2)
        x = copy.deepcopy(x0)
        interval = 0
        sol = None 

        # Currently only uses the first problem in the list, should be modified for all problems 
        # Can do separate problems in parallel with joblib 

        problem = self.problems[0]
        u0 = self.u0s[0]
        endtime = self.endtimes[0]
        for i in tqdm.tqdm(range(n_iterations)): 
            # call train_step
            inputs, outputs, coarsen_inputs_l, coarsen_inputs_r, x, sol, interval, flag = self.train_step(x, problem, u0, endtime, interval, sol)
            training_refine_inputs[i] = inputs 
            training_refine_outputs[i] = outputs
            
            if i != 0: 
                training_coarsen_inputs[2 * i - 2] = coarsen_inputs_l
                training_coarsen_inputs[2 * i - 1] = coarsen_inputs_r

                training_coarsen_outputs[2 * i - 2] = -training_refine_outputs[i-1]
                training_coarsen_outputs[2 * i - 1] = -training_refine_outputs[i-1]
                
            # if a flag is raised, reset to initial case and continue generating data
            if flag: 
                x = copy.deepcopy(x0)
                sol = None 
        combined_df = pd.DataFrame(np.c_[training_coarsen_inputs, training_coarsen_outputs]).dropna()
        training_coarsen_inputs = combined_df.iloc[:, :-1]
        training_coarsen_outputs = combined_df.iloc[:, -1]

        return training_refine_inputs, training_refine_outputs, training_coarsen_inputs, training_coarsen_outputs

    def train_step(self, x, problem, u0, endtime, old_interval, sol = None):
        """
        A single step of generating training data 

        Parameters: 
        ----------- 
        x: np.ndarray 
            Array of floats containing the mesh 

        problem: LinearAdvection 
            The problem which data is being generated for 

        u0: np.ndarray 
            Initial condition 

        endtime: float 
            problem endtime 

        old_interval: int 
            The interval chosen for refinement from the previous iteration 
            Used to generate the cell inputs for the coarsen reward 

        sol: np.ndarray 
            The solution on this grid calculated from the previous iteration 
            None for the first iteration of every restart 

        Returns: 
        --------
        training_inputs: np.ndarray 
            The inputs of the interval chosen for refinement in the current step 

        training_outputs: np.ndarray
            The reward assigned to the refinement of the current step 

        training_inputs_coarsen_l: np.ndarray
            The inputs of the left cell 
                assigned to the coarsen reward from the previous step 

        training_inputs_coarsen_r: np.ndarray
            The inputs of the right cell 
                assigned to the coarsen reward from the previous step 

        new_x: np.ndarray 
            The new (refined mesh) 

        new_sol: np.ndarray 
            The solution generated on the refined mesh 

        interval_choice: int
            The interval chosen for refinement 
            Used in the next step to find the inputs of the coarsen reward 
                generated in this step

        flag: bool 
            True if a stopping condition has been reached 
        """
        n_p = x.size 
        p = n_p / self.max_elements
        c = problem.c 

        if sol is None: 
            original_sol = problem.solve_linear_advection(u0(x), x, endtime)
            calculate_coarsen_values = False
        else: 
            original_sol = sol
            calculate_coarsen_values = True 

        gradients, delta_x = self.get_gradients(c, x, original_sol)
        
        avg_gradient = np.mean(gradients)

        interval_choice = np.random.randint(0, len(gradients))
        local_gradient = gradients[interval_choice]

        if interval_choice == 0:
            lg = gradients[interval_choice] 
        else: 
            lg = gradients[interval_choice-1]

        if interval_choice == len(gradients)-1:
            rg = gradients[interval_choice]
        else: 
            rg = gradients[interval_choice + 1]

        size = delta_x[interval_choice]

        average_size = np.mean(delta_x)

        training_inputs = [local_gradient, lg, rg, avg_gradient, size, average_size, p, c]

        if calculate_coarsen_values: 
            local_gradient_old_l = gradients[old_interval]
            local_gradient_old_r = gradients[old_interval + 1]

            if old_interval == 0:
                lg_old_l = gradients[old_interval] 
                lg_old_r = gradients[old_interval]
            else: 
                lg_old_l = gradients[old_interval-1]
                lg_old_r = gradients[old_interval]

            if old_interval == len(gradients)-2:
                rg_old_l = gradients[old_interval + 1]
                rg_old_r = gradients[old_interval + 1]
            else: 
                rg_old_l = gradients[old_interval + 1]
                rg_old_r = gradients[old_interval + 2]
            
            size_old_l = delta_x[old_interval]
            size_old_r = delta_x[old_interval + 1]

            training_inputs_coarsen_l = [local_gradient_old_l, lg_old_l, rg_old_l, avg_gradient, size_old_l, average_size, p, c]
            training_inputs_coarsen_r = [local_gradient_old_r, lg_old_r, rg_old_r, avg_gradient, size_old_r, average_size, p, c]

        else: 
            training_inputs_coarsen_l = np.full(8, np.nan)
            training_inputs_coarsen_r = np.full(8, np.nan)

        new_x = np.insert(x, interval_choice+1, 1/2* (x[interval_choice] + x[interval_choice + 1]))
        new_u0 = u0(new_x)

        problem.T = 0
        new_sol = problem.solve_linear_advection(new_u0, new_x, endtime, plotting=False)
        
        projected_original_sol = np.insert(original_sol, interval_choice+1, 1/2 * (original_sol[interval_choice] + original_sol[interval_choice + 1]))

        reward, flag = self.get_reward(new_sol, projected_original_sol, new_x, p)
        training_outputs = reward

        return training_inputs, training_outputs, training_inputs_coarsen_l, training_inputs_coarsen_r, new_x, new_sol, interval_choice, flag

    def get_reward(self, new_sol, original_sol, new_x, p): 
        """
        Obtains the reward for a given refinement 
        Format is R = (integrated change in solution - gamma_c * Increase in computational cost)

        Parameters: 
        -----------
        new_sol: np.ndarray 
            The solution from the refined set of x coordinates 

        original_sol: np.ndarray 
            The solution from the unrefined set of x coordinates 
        
        new_x: np.ndarray 
            The new set of x coordinates 

        p: int 
            The number of elements in the old set of x coordinates 

        Returns: 
        --------
        reward: float
            The calculated reward

        flag: bool 
            True if the computational budget has been met 
        """
        p_new = p + 1 / self.max_elements
        if p_new >= 1: 
            return -10, True
        R_comp = p_new ** 0.5 / (1 - p_new) - p ** 0.5 / (1-p_new)
        
        du = np.abs(new_sol - original_sol)
        delta_uh = simpson(du, x=new_x)
        R_uh = delta_uh
        
        reward = R_uh - self.gamma_c * R_comp

        return reward, False

    def get_gradients(self, c, x, sol):
        """
        Obtains the solution gradients and input spacing of a non-uniform grid
        c used for nonlinear advection problems to preserve upwinding 
        If c is positive, the first gradient should not be used 
        if c is negative, the last gradient should not be used 

        Parameters: 
        ----------- 
        c: float
            advection speed 
        x: np.ndarray 
            x-coordinates used for evaluation
        sol: np.ndarray: 
            The problem solution for gradient calculation 

        Return: 
        -------
        gradients: np.ndarray 
            The gradient of the sol with respect to x 
            The size is one less than the number of points 
        
        delta_x: np.ndarray 
            The spacing of the input x-coordinates 
            The size is one less than the number of points 
        """
        if c > 0: 
            delta_x = (x - np.roll(x, 1))[1:]
            delta_sol = (sol - np.roll(sol, 1))[1:]
        else: 
            delta_x = (np.roll(x, -1) - x)[:-1]
            delta_sol = (np.roll(sol, -1) - sol)[:-1] 

        gradients = delta_sol / delta_x 

        return gradients, delta_x

    def train_nn(self, refine_inputs, refine_outputs, coarsen_inputs, coarsen_outputs, plot_loss = False): 
        """
        Trains a neural network object based on given inputs and outputs 
        Calls the train method of neural_net.py
        Serializes and saves the neural networks as "NN_r" and "NN_c"
        
        Parameters:
        -----------
        refine_inputs: np.ndarray 
            The training inputs for refine 

        refine_outputs: np.ndarray 
            The training outputs for refine 

        coarsen_inputs: np.ndarray 
            The training inputs for coarsen 

        coarsen_outputs: np.ndarray 
            The training outputs for coarsen 

        plot_loss: bool 
            Whether to plot the training loss function 
            default: False
        """
        self.Q_r.train(refine_inputs, refine_outputs)
        self.Q_c.train(coarsen_inputs, coarsen_outputs)
        
        if plot_loss:
            pd.DataFrame(self.Q_r.model.loss_curve_).plot()
            plt.show()

            pd.DataFrame(self.Q_r.model.loss_curve_).plot()
            plt.show()

        # Serialize and save the neural network so we do not have to retrain
        with open("NN_r", "wb") as f: 
            pickle.dump(self.Q_r, f)

        with open("NN_c", "wb") as f: 
            pickle.dump(self.Q_c, f)
    
    def evaluate(self, x, problem, u0, endtime, max_iter = 50): 
        """
        Performs adaptive mesh refinement based on the learned value function 

        Parameters: 
        -----------
        x: np.ndarray
            unrefined array of coordinates 
        
        problem: LinearAdvection
            The fluid problem to evaluate

        u0: np.ndarray 
            The initial condition of the fluid problem 

        endtime: float 
            The endtime of the problem solution (static case)

        max_iter: int 
            The maximum number of iterations of AMR to perform 
            Defaults to 20 

        Returns: 
        --------
        new_x: np.ndarray
            The output array of refined x coordinates 
        
        final_solution: np.ndarray 
            The final solution on the refined x coordinates
        """
        new_x = copy.deepcopy(x) 
        for i in range(max_iter): 
            p = new_x.size / self.max_elements
            c = problem.c 

            print (f"iteration {i}")
            # solve the fluid problem on the current grid 
            problem.T = 0
            original_sol = problem.solve_linear_advection(u0(new_x), new_x, endtime, plotting=True)

            problem.plot(original_sol, new_x)
            gradients, delta_x = self.get_gradients(problem.c, new_x, original_sol)
            
            local_gradients = gradients
            avg_gradient = np.full_like(local_gradients, np.mean(gradients))

            lg = np.roll(gradients, -1)
            rg = np.roll(gradients, 1)

            size = delta_x
            average_size = np.full_like(size, np.mean(delta_x))

            # obtain the inputs of the neural network 
            inputs = np.c_[local_gradients, lg, rg, avg_gradient, size, average_size, np.full_like(size, p), np.full_like(size, c)]
            r_outputs = self.Q_r.predict(inputs)
            c_outputs = self.Q_c.predict(inputs)

            # Decide whether to refine or not 
            outputs_refine = r_outputs - 0.03
            outputs_coarsen = c_outputs
            outputs_nothing = np.zeros_like(c_outputs)

            outputs = np.c_[outputs_refine, outputs_nothing, outputs_coarsen]
            actions = np.argmax(outputs, axis = 1)

            temp_x = [new_x[0]]

            for i in range(actions.size): 
                if actions[i] == 0: 
                    # Refine 
                    temp_x.append(1/2 * (new_x[i] + new_x[i+1]))
                    temp_x.append(new_x[i+1])
                elif actions[i] == 1: 
                    # do nothing 
                    temp_x.append(new_x[i+1])
                elif actions[i] == 2: 
                    # coarsen 
                    if i == actions.size - 1: 
                        temp_x.append(new_x[i+1])
                    pass 

            if len(temp_x) == len(new_x): 
                return new_x, original_sol

            new_x = np.array(temp_x)

        return new_x, original_sol

    def load_NN(self, filename_r, filename_c): 
        """
        Load neural nets saved with the given filenames

        Parameters:
        -----------
        filename_r: 
            The filename to open for the refine neural net 

        filename_c: 
            The filename to open for the coarsen neural net 
        """
        with open(filename_r, "rb") as f: 
            self.Q_r = pickle.load(f)

        with open(filename_c, "rb") as f: 
            self.Q_c = pickle.load(f)

def generate_data(n_points, agent):
    """
    Generate a number of points for training
    Serializes the inputs and outputs for later use 

    Parameters: 
    -----------
    n_points: 
        The number of data points to generate 

    agent: 
        The RL agent to generate data for 

    Returns: 
    --------
    r_inputs: 
        The generated refine neural network inputs 

    r_outputs: 
        The generated refine neural network outputs 

    c_inputs: 
        The generated coarsen neural network inputs 

    c_outputs: 
        The generated coarsen neural network outputs 
    """
    r_inputs, r_outputs, c_inputs, c_outputs = agent.gen_training_data(np.linspace(0, 2, 5), n_points)
    print (c_outputs)

    with open("r_inputs", 'wb') as f: 
        pickle.dump(r_inputs, f)

    with open("r_outputs", "wb") as f: 
        pickle.dump(r_outputs, f)

    with open("c_inputs", 'wb') as f: 
        pickle.dump(c_inputs, f)

    with open("c_outputs", "wb") as f: 
        pickle.dump(c_outputs, f)

    return r_inputs, r_outputs, c_inputs, c_outputs

if __name__ == "__main__": 
    """
    Either train data and run the solution, or use serialized solutions
    """

    # Initialize RL agent
    rl = RL_agent(0.15, 
                  Q_r = NeuralNet(n_layers = 4, hidden_neurons=64, max_iter = 5000), 
                  Q_c = NeuralNet(n_layers = 4, hidden_neurons=64, max_iter = 5000))
    
    # Generate data for training the neural networks 
    #r_inputs, r_outputs, c_inputs, c_outputs = generate_data(20000, rl)

    # Load pretrained neural nets (if already trained)
    rl.load_NN("NN_r", "NN_c")

    # Train neural nets (if not yet trained)
    #rl.train_nn(r_inputs, r_outputs, c_inputs, c_outputs, plot_loss=True)

    # Use adaptive mesh refinement to solve a given problem
    new_x, sol = rl.evaluate(np.linspace(0, 2, 5), 
                             LinearAdvection(1, 0.8, 0), 
                             lambda x: 1-np.tanh(2*(1-4*(x-1/4))), 0.5)
    
    # print the number of points used 
    print (new_x.size)

    # plot the final solution 
    x_ex = np.linspace(0, 2, 100)
    u_ex = 1-np.tanh(2*(1-4*(x_ex-1/4 - 0.5)))

    plt.figure(1)
    plt.cla()
    plt.plot(x_ex, u_ex, color = 'black')
    plt.plot(new_x, sol, "o-")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()

    with open("new_x", "wb") as f: 
        pickle.dump(new_x, f)