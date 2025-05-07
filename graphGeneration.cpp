#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>

#include "nlohmann/json.hpp"

#include "json.h"
using json = nlohmann::json;

std::vector<int> must_nodes;
std::vector<std::vector<int>> adj_matrix;


int n = 5;          // number of nodes
double p = 0.4;     // graph density
bool must_have_nodes = true;


void generateErdosRenyiGraph(int n, double p, std::vector<std::vector<int>> &adj_matrix, std::mt19937 &generator) {
    std::uniform_real_distribution<double> ge(0,1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double randProb = ge(generator);

            if (i == j) {
                adj_matrix[i][j] = 0;
            } else if (randProb < p) {
                adj_matrix[i][j] = 1;
                adj_matrix[j][i] = 1;
            } else {
                adj_matrix[i][j] = 0;
                adj_matrix[j][i] = 0;
            }

        }
    }
}


std::vector<int> generateMustHaveNodes(int n, std::mt19937 &engine) {
    std::uniform_int_distribution<> cnt{1, (int)ceil(0.2*n)};
    int must_nodes_count = cnt(engine);

    std::vector<int> must_have_nodes(2*must_nodes_count+1);
    std::iota(begin(must_have_nodes), end(must_have_nodes)-1, 0);
    std::random_shuffle(begin(must_have_nodes), end(must_have_nodes));
    must_have_nodes.resize(must_nodes_count);

    return must_have_nodes;
}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "Pass two arguments! Number of nodes and density!" << std::endl;
        return 1;
    }

    n = std::stoi(argv[1]);
    p = std::stod(argv[2]);

    adj_matrix.resize(n);
    for (int i = 0; i < adj_matrix.size(); i++) {
        adj_matrix[i].resize(n);
    }

    std::random_device rd{};
    std::mt19937 generator{rd()};

    std::string instanceName = "n" + std::to_string(n)+ "p0" + std::to_string(int(p*10));
 
    generateErdosRenyiGraph(n, p, adj_matrix, generator);
    must_nodes = generateMustHaveNodes(n, generator);

    saveAdjToJSON(adj_matrix, n, must_have_nodes, must_nodes, instanceName);

    return 0;
}
