#include <iostream>
#include <string>
#include <fstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;


json readJSON(std::string name) {
    std::ifstream f(name);
    json data = json::parse(f);

    return data;
}


void saveAdjToJSON(std::vector<std::vector<int>> &adj_matrix, int n, bool must_have_nodes, std::vector<int> must_nodes, std::string instanceName) {
    json newJSON;
    for (int i = 0; i < n; i++) {
        std::string ii = std::to_string(i);
        newJSON[ii] = json::array();
        for (int j = 0; j < n; j++) {
            if (adj_matrix[i][j] == 1) {
                newJSON[ii].push_back(1);
            } else {
                newJSON[ii].push_back(0);
            }
           
        }
    }
    if (must_have_nodes) {
        newJSON["-1"] = json::array();
        for (int i = 0; i < must_nodes.size(); i++) {
            newJSON["-1"].push_back(must_nodes[i]);
        }
    }

    std::ofstream writer;
    std::string path = "graphs/" + instanceName + ".json";
    writer.open(path);
    writer << newJSON.dump(4);
}


void saveCover(std::vector<int> cover, std::string path) {
    json newJSON;

    newJSON = json::array();
    newJSON = cover;

    std::ofstream writer;
    writer.open(path);
    writer << newJSON.dump(4);
}