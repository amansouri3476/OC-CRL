from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.lp_solver import lp_solver_pulp, lp_solver_cvxpy, lp_solver_cvxpy_unconstrained, lp_solver_pulp_unconstrained
from scipy.optimize import linear_sum_assignment

def main():
    
    import slot_based_disentanglement.utils.general as utils
    import time
    
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(f"cuda device: {torch.cuda.current_device()}")
    device = torch.cuda.current_device()
    n_balls_list = np.arange(2,11)
    batch_size_list = [2, 16, 64, 128, 256]
    use_pulp = False

    
    for n_balls in n_balls_list:
        if use_pulp:
            time_recordings = {
                "MxS, lin_sum_assignment": [],
                "MxS, cvxpylayers": [],
                "MxS, pulp": [],
                "MxS^2, cvxpylayers": [],
                "MxS^2, pulp": [],
            }
        else:
            time_recordings = {
                "MxS, lin_sum_assignment": [],
                "MxS, cvxpylayers": [],
                "MxS^2, cvxpylayers": [],
            }

        n_b = n_balls
        n_s = n_b + 3

        # ------------------ MS with linear sum assignment ------------------ #
        for batch_size in batch_size_list:
            cost = torch.rand(batch_size, n_b, n_s, device=device)
            t0 = time.perf_counter()
            indices = np.array(
                list(
                    map(linear_sum_assignment, cost.detach().cpu().numpy())),
                    dtype=int
                    )
            time_recordings["MxS, lin_sum_assignment"].append(time.perf_counter()-t0)

            print(f"n_balls={n_balls}, batch_size={batch_size}, MS with linear sum assignment finished")

        # ------------------ MS with cvxpylayers ------------------ #
        cvxpylayer = lp_solver_cvxpy_unconstrained(n_b, n_s)
        for batch_size in batch_size_list:
            cost = torch.rand(batch_size, n_b, n_s, device=device)
            t0 = time.perf_counter()
            _ = cvxpylayer(cost.clone().detach())[0]
            time_recordings["MxS, cvxpylayers"].append(time.perf_counter()-t0)
            print(f"n_balls={n_balls}, batch_size={batch_size}, MS with cvxpylayers finished")

        # ------------------ MS with pulp ------------------ #
        if use_pulp:
            for batch_size in batch_size_list:
                cost = torch.rand(batch_size, n_b, n_s, device=device)
                t0 = time.perf_counter()
                _ = np.array(
                    list(
                        map(lp_solver_pulp_unconstrained, cost.detach().cpu().numpy())),
                        dtype=int
                        )
                time_recordings["MxS, pulp"].append(time.perf_counter()-t0)
                print(f"n_balls={n_balls}, batch_size={batch_size}, MS with pulp (not ILP) finished")


        # ------------------ MS^2 with cvxpylayers ------------------ #
        cvxpylayer = lp_solver_cvxpy(n_b, n_s)
        for batch_size in batch_size_list:
            cost = torch.rand(batch_size, n_b, n_s ** 2, device=device)
            t0 = time.perf_counter()
            _ = cvxpylayer(cost.clone().detach())[0]
            time_recordings["MxS^2, cvxpylayers"].append(time.perf_counter()-t0)
            print(f"n_balls={n_balls}, batch_size={batch_size}, MS^2 with cvxpylayers (constrained) finished")
        
        # ------------------ MS^2 with pulp ------------------ #
        if use_pulp:
            for batch_size in batch_size_list:
                cost = torch.rand(batch_size, n_b, n_s ** 2, device=device)
                t0 = time.perf_counter()
                _ = list(
                    map(lp_solver_pulp, cost.detach().cpu().numpy()))

                time_recordings["MxS^2, pulp"].append(time.perf_counter()-t0)
                print(f"n_balls={n_balls}, batch_size={batch_size}, MS^2 with pulp (not ILP) finished")

        
        # plot the time recordings over the batch size, and legend with batch size (#methods curves of time-#slot per each batch size)
        fig = plt.figure()
        styles = [".", "*", "^", "o", "x"]
        i = 0
        for method, recorded_time in time_recordings.items():
            plt.plot(batch_size_list, recorded_time, styles[i], label=method)
            plt.xlabel("batch size")
            plt.ylabel("time")
            i += 1
        
        plt.legend()
        plt.show()
        plt.title(f"#balls={n_b}, #slots={n_s}, gpu={torch.cuda.is_available()}, #cores={4}")
        plt.savefig(f"n_balls_{n_b}, gpu={torch.cuda.is_available()}, #cores={4}.png")
        plt.cla()

        # # plot the time recordings over the number of slots, and legend with batch size (#methods curves of time-#slot per each batch size)
        # fig = plt.figure()
        # styles = [".", "*", "^", "o", "x"]
        # i = 0
        # for j, batch_size in enumerate(batch_size_list):
        #     plt.plot(batch_size_list, recorded_time, styles[i], label=method)
        #     plt.xlabel("batch size")
        #     plt.ylabel("time")
        #     i += 1
        
        # plt.legend()
        # plt.show()
        # plt.title(f"#balls={n_b}, #slots={n_s}, gpu={torch.cuda.is_available()}, #cores={4}")
        # plt.savefig(f"_n_balls_{n_b}, gpu={torch.cuda.is_available()}, #cores={4}.png")
        # plt.cla()

if __name__ == "__main__":
    main()
