// g++ geneticOpenMP.cpp -o open2 -fopenmp -O3 -std=c++11

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <omp.h>

#include "nlohmann/json.hpp"
#include "json.h"

#include "string.h"

using json = nlohmann::json;


int n = 5;          // number of nodes
double p = 0.4;     // graph density
std::string graphName = "n5p04";

bool must_have_nodes = true;

std::vector<int> must_nodes;
std::vector<std::vector<int>> adj_matrix;

int max_population = 30;
float mutation_rate = 0.1;
int generations = 1;
int max_generations = 10;

int max_threads;

std::vector<std::vector<int>> initialization(int n, int max_population, std::mt19937 engine, bool must_have_nodes, std::vector<int> must_nodes) {
    std::uniform_int_distribution<int> dist{0,1};
    
    std::vector<std::vector<int>> population;
    population.resize(max_population);

    for (int i = 0; i < max_population; i++){   // iterating through chromosomes
        for (int j = 0; j < n; j++){            // iterating through genes
            // setting constant genes if there are vertices to include in the cover
            if (must_have_nodes) {
                if (std::find(must_nodes.begin(), must_nodes.end(), j) != must_nodes.end()) {
                    population[i].push_back(1);
                } else {
                    population[i].push_back(dist(engine));
                }

            } else {
                population[i].push_back(dist(engine));
            }    
        }
    }
    return population;
}


double fitness(std::vector<int> chromosome, std::vector<std::vector<int>> &adj_matrix, int n, int number_of_edges) {

    std::vector<int> fitness_matrix(n * n, 0);

    double fitness_count = 0;    
    double penalty = 0;
    for(int gene = 0; gene < chromosome.size(); gene++) {
        if (chromosome[gene] == 1) {
            for (int j = 0; j < n; j++)  {
                if (adj_matrix[gene][j] == 1) {
                    if (fitness_matrix[gene * n + j] == 1) {
                        penalty += 1;
                    } else {
                        fitness_count += 1;
                    }
                    fitness_matrix[gene * n + j] = 1;
                    fitness_matrix[j * n + gene] = 1;
                }
            }
        } 
    }

    fitness_count = fitness_count/number_of_edges*100 - penalty / number_of_edges;
    return fitness_count;
}


std::vector<int> crossover(std::vector<int> parentA, std::vector<int> parentB, std::mt19937 engine) {
    std::vector<int> child;
    int size = parentA.size();

    std::uniform_int_distribution<int> mid(0, size-1);

    int midpoint = mid(engine);

    for (int gene = 0; gene < size; gene++) {
        if (gene > midpoint) {
            child.push_back(parentB[gene]);
        } else {
            child.push_back(parentA[gene]);
        }
    }

    return child;
    
}

std::vector<int> mutate(std::vector<int> child, double mutation_probability, std::mt19937 engine, bool must_have_nodes, std::vector<int> must_nodes) {

    std::uniform_real_distribution<> mut(0,1);
    double mutation_random;
    std::uniform_int_distribution<int> gen_mut{0,1};

    for (int gene = 0; gene < child.size(); gene++) {
        if (must_have_nodes && std::find(must_nodes.begin(), must_nodes.end(), gene) != must_nodes.end()) {
            continue;
        }
        mutation_random = mut(engine);
        if (mutation_random < mutation_probability) {
            child[gene] = gen_mut(engine);
        }
    }
    return child;
}

std::vector<int> genetic(std::vector<std::vector<int>> &adj_matrix, int n, bool must_have_nodes, std::vector<int> must_nodes, std::mt19937 &engine) {

    // counting edges in graph for later use in fitness function
    int number_of_edges = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (adj_matrix[i][j] == 1) {
                number_of_edges++;
            }
        }
    }

    // first population *************************
    std::vector<std::vector<int>> population;
    population = initialization(n, max_population, engine, must_have_nodes, must_nodes);

    // fitness **********************************
    std::vector<double> population_fitness;
    population_fitness.resize(max_population);

    // looping till all generations are generated
    
    for (int gen = 0; gen < max_generations-1; gen++) {

        // calculating fitness scores
        #pragma omp parallel for default(none) shared(population, population_fitness, adj_matrix, number_of_edges, n, max_population)
        for (int i = 0; i < max_population; i++) {
            population_fitness[i] = fitness(population[i], adj_matrix, n, number_of_edges);
        }

        // selection & reproduction **********************************

        std::vector<std::vector<int>> new_population;
        new_population.reserve(max_population);


        #pragma omp parallel 
        {

            std::vector<std::vector<int>> local_population;

            // local generator
            std::random_device rd_local;
            std::mt19937 local_engine(rd_local() + omp_get_thread_num());

            std::discrete_distribution<> d(population_fitness.begin(), population_fitness.end());

            #pragma omp for nowait
            for (int chrom = 0; chrom < population.size(); chrom++) {

                std::vector<int> parentA = population[d(local_engine)];
                std::vector<int> parentB = population[d(local_engine)];
    
                std::vector<int> child = crossover(parentA, parentB, local_engine);
                child = mutate(child, mutation_rate, local_engine, must_have_nodes, must_nodes);
    
                local_population.push_back(child);
            }

            #pragma omp critical
            {
                new_population.insert(new_population.end(), local_population.begin(), local_population.end());
            }


        }

        population = new_population;

    }

    // calculating fitness scores for the last population
    #pragma omp parallel for default(none) shared(population, population_fitness, adj_matrix, number_of_edges, n, max_population)
    for (int i = 0; i < max_population; i++) {
        population_fitness[i] = fitness(population[i], adj_matrix, n, number_of_edges);
    }

    // picking chromosome with highest fitness score
    std::vector<double> max_fit = {-1, -1};
    for (int fit = 0; fit < population_fitness.size(); fit++) {
        if (population_fitness[fit] > max_fit[0]) {
            max_fit[0] = population_fitness[fit];
            max_fit[1] = fit;
        } 
    }

    // changing 'chromosome notation' to vertices
    int max_fit_int = (int)max_fit[1];
    std::vector<int> min_cover;
    for (int i = 0; i < population[max_fit_int].size(); i++) {
        if (population[max_fit_int][i] == 1) {
            min_cover.push_back(i);
        }
    }
 
    return min_cover;
}



int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cout << "Pass 5 arguments! Number of nodes, density, population, generations and mutation rate!" << std::endl;
        return 1;
    }

    max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);


    n = std::stoi(argv[1]);
    p = std::stod(argv[2]);
    max_population = std::stoi(argv[3]);
    max_generations = std::stoi(argv[4]);
    mutation_rate = std::stof(argv[5]);

    graphName = "n" + std::to_string(n) + "p0" + std::to_string((int)(p * 10));
    
    std::random_device rd{};
    std::mt19937 generator{rd()};

    adj_matrix.resize(n);
    for (int i = 0; i < adj_matrix.size(); i++) {
        adj_matrix[i].resize(n);
    }

    std::string graphPath = "graphs/" + graphName + ".json";
    json data = readJSON(graphPath);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++) {
            adj_matrix[i][j] = (int)data[std::to_string(i)][j];
        }
    }

    must_nodes.clear();

    for (int i = 0; i < (int)data["-1"].size(); i++) {
        must_nodes.push_back((int)data["-1"][i]);
    }



    std::vector<int> minCover;
    json output;


    auto begin = std::chrono::high_resolution_clock::now();  

    minCover = genetic(adj_matrix, n, must_have_nodes, must_nodes, generator);

    auto end = std::chrono::high_resolution_clock::now();
    int time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();

    output["nodes"] = n;
    output["density"] = p;
    output["time"] = time;


    std::ofstream writer;
    writer.open("outputs/" + graphName + ".json");
    writer << output.dump(4);
    
    std::string coverFile = "covers/" + graphName + ".json"; 
    saveCover(minCover, coverFile);

    return 0;
}
