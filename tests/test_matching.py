import unittest
import torch
import numpy as np
import scipy
from scipy import optimize
from pytorch_lightning import seed_everything


class TestMatching(unittest.TestCase):
    def setUp(self) -> None:
        """Runs before each test method"""
        
        # Define a couple of matrices of pairwise loss and concat them to make a batch
        pairwise_costs_list = []
        indices_list = []
        costs_list = []
        num_slots = 5
        n_mech = 4
        pairwise_costs_list.append(
            torch.tensor(
                np.array(
                    [
                    [1, 4, -5, -2, 7],
                    [4, 6, 1, -3, 5],
                    [3, -1, -2, 4, 6],
                    [2, 7, 9, -11, -1],
                    ]
                )
                
            )
        )
        indices_list.append(
            torch.tensor(
                np.array(
                    [
                        [0, 2],
                        [1, 0],
                        [2, 1],
                        [3, 3],
                    ]
                )
            )                    
        )
        costs_list.append(
            torch.tensor(
                np.array(
                    [
                        [-5],
                        [4],
                        [-1],
                        [-11],
                    ]
                )
            )
        )

        pairwise_costs_list.append(
            torch.tensor(
                np.array(
                    [
                        [-1, -4, -5, 2, -7],
                        [2, 2, 1, -3, -5],
                        [-1, -1, 2, 6, 9],
                        [3, -4, 2, 1, -10],
                    ]
                )
            )
        )
        indices_list.append(
            torch.tensor(
                np.array(
                    [
                        [0, 2],
                        [1, 3],
                        [2, 0],
                        [3, 4],
                    ]
                )
            )
        )
        costs_list.append(
            torch.tensor(
                np.array(
                    [
                        [-5],
                        [-3],
                        [-1],
                        [-10],
                    ]
                )
            )
        )

        pairwise_costs_list.append(
            torch.tensor(
                np.array(
                    [
                        [2, -7, -2, -1, -4],
                        [0, 4, -3, -2, -7],
                        [-2, 1, -1, 6, 3],
                        [-1, -4, 2, 10, -5],
                    ]
                )
            )
        )
        indices_list.append(
            torch.tensor(
                np.array(
                    [
                        [0, 1],
                        [1, 2],
                        [2, 0],
                        [3, 4],
                    ]          
                )      
            )
        )
        costs_list.append(
            torch.tensor(
                np.array(
                    [
                        [-7],
                        [-3],
                        [-2],
                        [-5],
                    ]
                )
            )
        )
        self.pairwise_costs = torch.stack(pairwise_costs_list, dim=0)
        self.indices = torch.stack(indices_list, dim=0)
        self.costs = torch.stack(costs_list, dim=0)

    def test_matching(self):

        indices = torch.tensor(
            list(map(scipy.optimize.linear_sum_assignment, self.pairwise_costs)))
        transposed_indices = torch.permute(indices, dims=(0, 2, 1))

        actual_costs = torch.gather(self.pairwise_costs, 2, transposed_indices[:,:,1].unsqueeze(-1)).float()

        print(transposed_indices)
        print(self.indices)

        print('-----------\n')
        print(actual_costs)
        print(self.costs)

        self.assertEqual(torch.eq(transposed_indices, self.indices).all(), True)
        self.assertEqual(torch.eq(actual_costs, self.costs).all(), True)


if __name__ == "__main__":
    unittest.main()