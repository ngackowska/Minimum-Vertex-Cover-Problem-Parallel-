#include "nlohmann/json.hpp"

void load_graph(const std::string& path, int* adj_matrix, int n, int*& must_nodes, int& must_count);


void save_data(int n, double p, int time, std::string graphName, int out_size, int* result);