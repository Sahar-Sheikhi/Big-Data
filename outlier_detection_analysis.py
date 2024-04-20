import math
import sys
import time
import os
from pyspark import SparkContext, SparkConf


# *************** EXACT ALGORITHM ***************

def ExactOutliers(InputPoints: list, D: float, M: int, K: int):
    """
    Implement the Exact Algorithm for outlier detection.

    Args:
    InputPoints (list): List of InputPoints represented as tuples (x, y).
    D (float): Threshold distance.
    M (int): Maximum number of neighbors.
    K (int): Number of first outliers to return.

    Returns:
    tuple: A tuple containing the number of outliers found and a list of top K outliers.
    """

    # Function to compute distance between two InputPoints
    def euclidean_distance(p1, p2):
        """
        Compute the Euclidean distance between two InputPoints.

        Args:
        p1 (tuple): First point represented as a tuple (x, y).
        p2 (tuple): Second point represented as a tuple (x, y).

        Returns:
        float: Euclidean distance between the two InputPoints with avoiding using sqrt for better runtime.
        """
        return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

    # Precompute the number of InputPoints within the threshold distance for each point
    D_power2 = D * D  # used D * D instead of using sqrt in Euclidean distance
    points_within_distance = [sum(1 for q in InputPoints
                                  if euclidean_distance(p, q) <= D_power2)
                              for p in InputPoints]

    # Identify outliers based on the precomputed number of nearby points
    outliers = []
    for p, B_p in zip(InputPoints, points_within_distance):
        if B_p <= M:
            outliers.append((p, B_p))

    # Sort outliers by the number of nearby points
    outliers.sort(key=lambda x: x[1])  # Sort outliers by |B_S(p, D)|

    # Count the number of outliers and select the top K outliers
    num_outliers = len(outliers)

    # The top K outliers
    k_first_outliers = outliers[:K]

    return num_outliers, k_first_outliers


# *************** APPROXIMATE ALGORITHM ***************
# Perform stepA
def stepA(inputPoints, lambda_):
    """
    Perform Step A of the MRApproxOutliers algorithm.

    Args:
    inputPoints: RDD containing input InputPoints.
    lambda_ (float): The side length of a cell.

    Returns:
    RDD: An RDD containing cell identifiers and their sizes.
    """
    # Map InputPoints to cells and count points in each cell
    stepA_resultRDD = (inputPoints
                       .map(lambda point: ((int(point[0] // lambda_),
                                            int(point[1] // lambda_)), 1))
                       .reduceByKey(lambda x, y: x + y))  # Aggregate cells and define cells' sizes

    return stepA_resultRDD


def stepB(stepA_rdd):
    """
    Perform Step B of MRApproxOutliers.

    Args:
    stepA_rdd: RDD containing cells identifiers and cell size after Step A.

    Returns:
    stepB_rdd: RDD containing cells information after Step B.
    (a tuple of two tuples, first one is cell identifier and second is a tuple of 3 including cell size,N3,N7)
    """

    def calculate_N3_N7(cell_dict, cell_p):
        """
        Count the neighbors of a cell (C_p).

        Args:
        cell_dict (dict): Dictionary containing cell identifiers as keys and their sizes as values.
        cell_p (tuple): Tuple containing the cell identifier and its size.

        Returns:
        tuple: Updated tuple containing the cell identifier and a tuple with its size, N3 count, and N7 count.

        """

        # Extract cell identifier and size from cell_p
        cell_id, cell_size = cell_p
        i, j = cell_id

        # Initialize N3 count
        N3 = 0

        # list to store neighbors within a 3x3 grid
        R3_cells = []

        # Count neighbors within a 3x3 grid
        for row in range(i - 1, i + 2):
            for col in range(j - 1, j + 2):
                if (row, col) in cell_dict:
                    N3 += cell_dict[(row, col)]  # Increment N3 count
                    R3_cells.append((row, col))  # Record the R3 neighbor's identifiers

        # Initialize N7 count (includes neighbors within a 7x7 grid, excluding the 3x3 grid)
        N7 = N3

        # Count additional neighbors within a 7x7 grid, excluding the 3x3 grid
        for row in range(i - 3, i + 4):
            for col in range(j - 3, j + 4):
                if (row, col) in cell_dict and (row, col) not in R3_cells:
                    N7 += cell_dict[(row, col)]  # Increment N7 count

        # Return the cell identifier along with its size, N3 count, and N7 count
        return (i, j), (cell_size, N3, N7)

    # Convert list of tuples (cell id and cell size) to dictionary
    celldict = {key: value for key, value in stepA_rdd.collect()}

    # Apply count_neighbors to each cell in stepA_rdd to add N3 and N7 information for each cell
    stepB_rdd = stepA_rdd.map(lambda cell: calculate_N3_N7(celldict, cell))

    return stepB_rdd


def MRApproxOutliers(inputPoints, D: float, M: int, K: int):
    """
    Perform the MRApproxOutliers algorithm.

    Args:
    inputPoints: RDD containing input InputPoints.
    D (float): Threshold distance.
    M (int): Threshold number of neighbors.
    K (int): Number of first outliers to return.

    Returns:
    tuple: A tuple containing the number of sure outliers,
    number of uncertain points, and The first 𝐾 non-empty cells.
    """
    # Calculate the side length of a cell
    lambda_ = D / (2 * (2 ** 0.5))

    # Perform Step A: Map points to cells and identify cells sizes
    cells_id_size_info = stepA(inputPoints, lambda_)

    # Perform Step B: Calculate N3 and N7 for each cell
    cells_N3_N7_info = stepB(cells_id_size_info)

    # Filter sure outliers
    sure_outliers_rdd = cells_N3_N7_info.filter(lambda cell: cell[1][2] <= M)

    # Filter uncertain points
    uncertain_points_rdd = cells_N3_N7_info.filter(lambda cell: cell[1][1] <= M < cell[1][2])

    # Counting the sure outliers and uncertain points
    sure_outliers = sure_outliers_rdd.map(lambda cell: cell[1][0]).sum()
    uncertain_points = uncertain_points_rdd.map(lambda cell: cell[1][0]).sum()

    # Calculating the first 𝐾 non-empty cells
    k_non_empty_cells = (cells_N3_N7_info
                         .map(lambda cell: (cell[1][0], cell))  # Assuming cell[1][0] contains the cell size
                         .sortByKey()
                         .map(lambda x: x[1])
                         .take(K))

    return sure_outliers, uncertain_points, k_non_empty_cells


def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python G024HW1 <data_path> <D> <M> <K> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('OUTLIER DETECTION')
    sc = SparkContext(conf=conf)

    # INPUT READING
    data_path = sys.argv[1]

    D = sys.argv[2]
    assert D.replace('.', '').isdigit(), "D must be a float"
    D = float(D)

    M = sys.argv[3]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    K = sys.argv[4]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    L = sys.argv[5]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # Read input file and subdivide it into L partitions
    rawData = sc.textFile(data_path)
    assert os.path.isfile(data_path), "File or folder not found"

    inputPoints = (rawData.map(lambda line: tuple(map(float, line.split(','))))
                   .repartition(numPartitions=L).cache())

    # Print the input file name and Parameters Values
    print(f"{os.path.basename(data_path)} D={D} M={M} K={K} L={L}")

    # Print total number of InputPoints
    Num_of_points = inputPoints.count()
    print("Number of InputPoints:", Num_of_points)

    # Perform ExactOutliers if the number of InputPoints is small
    if Num_of_points <= 200000:
        listOfPoints = inputPoints.collect()
        ExactOutliers_start_time = time.time()
        num_outliers, outliers = ExactOutliers(listOfPoints, D, M, K)
        print("Number of Outliers =", num_outliers)
        for outlier in outliers:
            print("Point:", outlier[0])

        ExactOutliers_end_time = time.time()

        print(f"Running time of ExactOutliers: {int((ExactOutliers_end_time - ExactOutliers_start_time) * 1000)} ms")

    # Perform MRApproxOutliers
    MRApproxOutliers_start_time = time.time()
    sure_outliers, uncertain_points, cells_info = MRApproxOutliers(inputPoints, D, M, K)
    MRApproxOutliers_end_time = time.time()
    print("Number of sure outliers:", sure_outliers)
    print("Number of uncertain InputPoints:", uncertain_points)
    # Print cells
    for cell in cells_info:
        print("Cell:", cell[0], "Size:", cell[1][0])
    print(f"Running time of MRApproxOutliers:"
          f" {int((MRApproxOutliers_end_time - MRApproxOutliers_start_time) * 1000)} ms")

    # Stop SparkSession
    sc.stop()


if __name__ == "__main__":
    main()
