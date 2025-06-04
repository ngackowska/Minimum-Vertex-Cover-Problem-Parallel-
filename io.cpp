#include "io.h"
#include <fstream>

using json = nlohmann::json;

void load_graph(const std::string& path, int* adj_matrix, int n, int*& must_nodes, int& must_count) {

    std::ifstream f(path);
    json data = json::parse(f);


    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            adj_matrix[i * n + j] = (int)data[std::to_string(i)][j];


    must_count = (int)data["-1"].size();
    must_nodes = new int[must_count];

    for (int i = 0; i < must_count; i++)
        must_nodes[i] = data["-1"][i];
}


void save_data(int n, double p, int time, std::string graphName, int out_size, int* result) {
    json output;


    output["nodes"] = n;
    output["density"] = p;
    output["time"] = time;


    std::ofstream writer;
    writer.open("outputs/" + graphName + ".json");
    writer << output.dump(4);

    json cover;
    for (int i = 0; i < out_size; i++) cover.push_back(result[i]);
    std::ofstream cover_file("covers/" + graphName + ".json");
    cover_file << cover.dump(4);
}