// g++ geneticOpenMP.cpp -o open6 -fopenmp -O3 -std=c++11
// ./open4 6 0.6 10 30 0.1

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

// 
int* must_nodes = nullptr;
int** adj_matrix = nullptr;
int must_nodes_count;

int max_population = 30;
float mutation_rate = 0.1;
int generations = 1;
int max_generations = 10;

int max_threads;


int** allocate_matrix(int rows, int cols) {
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
        std::fill(matrix[i], matrix[i] + cols, 0);
    }
    return matrix;
}

void deallocate_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) delete[] matrix[i];
    delete[] matrix;
}




int** initialization(int n, int max_population, std::mt19937 engine, bool must_have_nodes, int* must_nodes, int must_nodes_size) {
    std::uniform_int_distribution<int> dist{0,1};
    

    int** population = allocate_matrix(max_population, n);


    for (int i = 0; i < max_population; i++){   // iterating through chromosomes

        for (int j = 0; j < n; j++){            // iterating through genes

           bool is_must = false;
            if (must_have_nodes) {
                for (int k = 0; k < must_nodes_count; ++k) {
                    if (must_nodes[k] == j) {
                        is_must = true;
                        break;
                    }
                }
            }
            population[i][j] = is_must ? 1 : dist(engine);




        }
    }
    return population;
}


double fitness(int* chromosome, int** adj_matrix, int n, int number_of_edges) {

    int** fitness_matrix = allocate_matrix(n, n);



    double fitness_count = 0;    
    double penalty = 0;


    for(int gene = 0; gene < n; gene++) {
        if (chromosome[gene] == 1) {
            for (int j = 0; j < n; j++) {
                if (adj_matrix[gene][j] == 1) {
                    if (fitness_matrix[gene][j] == 1) {
                        penalty += 1;
                    } else {
                        fitness_count += 1;
                    }
                    fitness_matrix[gene][j] = 1;
                    fitness_matrix[j][gene] = 1;
                }
            }
        } 
    }



    deallocate_matrix(fitness_matrix, n);

    fitness_count = fitness_count/number_of_edges*100 - penalty / number_of_edges;
    return fitness_count;
}



int* crossover(int* parentA, int* parentB, int size, std::mt19937& engine) {
    int* child = new int[size];
    std::uniform_int_distribution<int> mid(0, size - 1);
    int midpoint = mid(engine);
    for (int gene = 0; gene < size; gene++) {
        child[gene] = (gene > midpoint) ? parentB[gene] : parentA[gene];
    }
    return child;
}


void mutate(int* child, int size, double mutation_probability, std::mt19937& engine, bool must_have_nodes, int* must_nodes, int must_nodes_count) {
    std::uniform_real_distribution<> mut(0, 1);
    std::uniform_int_distribution<int> gen_mut(0, 1);

    for (int gene = 0; gene < size; gene++) {
        if (must_have_nodes) {
            bool is_must = false;
            for (int k = 0; k < must_nodes_count; ++k) {
                if (must_nodes[k] == gene) {
                    is_must = true;
                    break;
                }
            }
            if (is_must) continue;
        }

        if (mut(engine) < mutation_probability) {
            child[gene] = gen_mut(engine);
        }
    }
}




int* genetic(int** adj_matrix, int n, bool must_have_nodes, int* must_nodes, int must_nodes_count, std::mt19937 &engine, int& out_size) {

    // counting edges in graph for later use in fitness function
    int number_of_edges = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (adj_matrix[i][j] == 1) {
                number_of_edges++;
            }
        }
    }


    int** population = initialization(n, max_population, engine, must_have_nodes, must_nodes, must_nodes_count);
    double* population_fitness = new double[max_population];


    // looping till all generations are generated
    
    for (int gen = 0; gen < max_generations-1; gen++) {

        // calculating fitness scores
        #pragma omp parallel for default(none) shared(population, population_fitness, adj_matrix, number_of_edges, n, max_population)
        for (int i = 0; i < max_population; i++) {
            population_fitness[i] = fitness(population[i], adj_matrix, n, number_of_edges);
        }

        // selection & reproduction **********************************

        int** new_population = allocate_matrix(max_population, n);


        #pragma omp parallel 
        {


            // local generator
            std::random_device rd_local;
            std::mt19937 local_engine(rd_local() + omp_get_thread_num());

            std::discrete_distribution<> d(population_fitness, population_fitness + max_population);


            // nowait
            #pragma omp for 
            for (int chrom = 0; chrom < max_population; chrom++) {

                int* parentA = population[d(local_engine)];
                int* parentB = population[d(local_engine)];
    
                int* child = crossover(parentA, parentB, n, local_engine);
                mutate(child, n, mutation_rate, local_engine, must_have_nodes, must_nodes, must_nodes_count);
    
                std::copy(child, child + n, new_population[chrom]);
                delete[] child;
            }


        }

        for (int i = 0; i < max_population; i++) delete[] population[i];
        delete[] population;
        population = new_population;

    }

    // calculating fitness scores for the last population
    #pragma omp parallel for default(none) shared(population, population_fitness, adj_matrix, number_of_edges, n, max_population)
    for (int i = 0; i < max_population; i++) {
        population_fitness[i] = fitness(population[i], adj_matrix, n, number_of_edges);
    }


    // picking chromosome with highest fitness score
    int best_idx = 0;
    double best_fit = population_fitness[0];
    for (int fit = 0; fit < max_population; fit++) {
        if (population_fitness[fit] > best_fit) {
            best_fit = population_fitness[fit];
            best_idx = fit;
        }
    }


    int* result = new int[n];
    out_size = 0;
    for (int i = 0; i < n; i++) {
        if (population[best_idx][i] == 1) {
            result[out_size++] = i;
        }
    }

    for (int i = 0; i < max_population; i++) delete[] population[i];
    delete[] population;
    delete[] population_fitness;

    return result;


}


int main(int argc, char *argv[]) {

    if (argc != 7) {
        std::cout << "Pass 6 arguments! Number of nodes, density, population, generations, mutation rate and number of threads!" << std::endl;
        return 1;
    }

    max_threads = omp_get_max_threads();
    


    n = std::stoi(argv[1]);
    p = std::stod(argv[2]);
    max_population = std::stoi(argv[3]);
    max_generations = std::stoi(argv[4]);
    mutation_rate = std::stof(argv[5]);
    max_threads = std::stoi(argv[6]);

    omp_set_num_threads(max_threads);

    graphName = "n" + std::to_string(n) + "p0" + std::to_string((int)(p * 10));
    
    std::random_device rd{};
    std::mt19937 generator{rd()};

    std::string graphPath = "graphs/" + graphName + ".json";
    json data = readJSON(graphPath);


    adj_matrix = allocate_matrix(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            adj_matrix[i][j] = (int)data[std::to_string(i)][j];


    must_nodes_count = (int)data["-1"].size();
    must_nodes = new int[must_nodes_count];
    must_have_nodes = must_nodes_count > 0;

    for (int i = 0; i < must_nodes_count; i++)
        must_nodes[i] = data["-1"][i];
    

    auto begin = std::chrono::high_resolution_clock::now();  

    int result_size;
    int* result = genetic(adj_matrix, n, must_have_nodes, must_nodes, must_nodes_count, generator, result_size);

    auto end = std::chrono::high_resolution_clock::now();
    int time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();


    json output;


    output["nodes"] = n;
    output["density"] = p;
    output["time"] = time;


    std::ofstream writer;
    writer.open("outputs/" + graphName + ".json");
    writer << output.dump(4);

    json cover;
    for (int i = 0; i < result_size; i++) cover.push_back(result[i]);
    std::ofstream cover_file("covers/" + graphName + ".json");
    cover_file << cover.dump(4);


    delete[] result;
    delete[] must_nodes;
    deallocate_matrix(adj_matrix, n);

    return 0;
}
