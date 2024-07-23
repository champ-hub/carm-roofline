#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <time.h>
#include "../dbi_carm_roi.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " matrix.mtx num_iterations" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }
    int num_iterations = std::stoi(argv[2]);

    int rows, cols, nonzeros;
    std::string line;

    // Read the first non-comment line as the header
    while (getline(file, line)) {
        if (line[0] != '%') {
            std::istringstream iss(line);
            iss >> rows >> cols >> nonzeros;
            break;
        }
    }

    // Define A as a row-major sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nonzeros);

    // Read the non-zero entries
    int row, col;
    double value;
    while (file >> row >> col >> value) {
        // Adjust from 1-based to 0-based index
        triplets.push_back(Eigen::Triplet<double>(row - 1, col - 1, value));
    }

    file.close();
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd x = Eigen::VectorXd::Random(cols);
    Eigen::VectorXd y(cols);

    std::chrono::duration<double> total_duration(0);
    std::vector<double> times;
    
    CARM_roi_begin();
    // Perform the SpMV multiple times
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        y = A * x;
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = finish - start;
        times.push_back(elapsed.count());
        total_duration += elapsed;
    }
    CARM_roi_end();

    double avg_ms = total_duration.count() * 1000.0 / num_iterations;

    double average_time = total_duration.count() / num_iterations;
    double nz_throughput = double(nonzeros) / avg_ms / 1.0e6;

    std::cout << "fp" << sizeof(double) * 8 << ": "
              << 2 * nonzeros << " total theoretical flops | "
	      << avg_ms << " avg ms | "
              << 2 * nz_throughput << " theoretical GFLOP/s\n";

    std::cout << "Total time for " << num_iterations << " iterations: " << total_duration.count() << " seconds.\n";
    std::cout << "Average time per iteration: " << average_time << " seconds.\n";
    std::cout << "Result vector y has size: " << y.size() << ".\n";

    return 0;
}
