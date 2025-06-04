
//  g++ -O3 -c io.cpp -o io.o
// nvcc -O3 -c geneticCUDA2.cu -o cudaTest.o
// g++ cudaTest.o io.o -o geneticCUDATest -lcudart -L/usr/local/cuda/lib64

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "json.h"
#include <ctime>
#include "string.h"

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "io.h"

int block_size = 256;

int n = 5;          // number of nodes
double p = 0.4;     // graph density
std::string graphName = "n5p04";

bool must_have_nodes = true;
int must_nodes_count = 0;

int max_population = 30;
float mutation_rate = 0.1;
int generations = 1;
int max_generations = 10;

int max_threads;



__device__ int get_index(int row, int col, int n) {
    return row * n + col;
}

__global__ void init_curand_states(curandState* states, unsigned long seed, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initialize_population_kernel(
    int* population, int n, int max_population,
    int* must_nodes, int must_nodes_count,
    bool must_have_nodes, curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_population) return;

    curandState local_state = states[idx];
    int* chrom = &population[idx * n];

    for (int gene = 0; gene < n; ++gene) {
        bool is_must = false;

        if (must_have_nodes) {
            for (int i = 0; i < must_nodes_count; ++i) {
                if (must_nodes[i] == gene) {
                    is_must = true;
                    break;
                }
            }
        }

        int bit = curand(&local_state) % 2;
        chrom[gene] = is_must ? 1 : bit;
    }

    states[idx] = local_state;
}



__global__ void fitness_kernel(
    int* population, float* fitness_scores,
    int* adj_matrix, int n, int number_of_edges, int max_population
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_population) return;

    int* chromosome = &population[idx * n];
    
    float fitness_count = 0.0f;
    float penalty = 0.0f;

    
    for (int gene = 0; gene < n; gene++) {
        if (chromosome[gene] == 1) {
            for (int j = 0; j < n; j++) {
                if (adj_matrix[get_index(gene, j, n)] == 1) {
                    if (gene < j) {
                        fitness_count += 1.0f;
                    } else if (gene > j && chromosome[j] == 1) {
               
                        penalty += 1.0f;
                    }
                }
            }
        }
    }

    fitness_scores[idx] = (fitness_count / number_of_edges) * 100.0f - (penalty / number_of_edges);
}


__global__ void reproduce_kernel(int* population, int* new_population, float* fitnesses, int n, int pop_size, float mutation_rate, int* must_have_nodes, int must_nodes_count, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];

    int parentA_idx = curand(&local_state) % pop_size;
    int parentB_idx = curand(&local_state) % pop_size;
    int* parentA = &population[parentA_idx * n];
    int* parentB = &population[parentB_idx * n];
    int* child = &new_population[idx * n];

    for (int i = 0; i < n; i++) {
        child[i] = (curand_uniform(&local_state) > 0.5f) ? parentA[i] : parentB[i];
        if (curand_uniform(&local_state) < mutation_rate) {
            child[i] = 1 - child[i];
        }
    }

    for (int i = 0; i < must_nodes_count; i++) {
        child[must_have_nodes[i]] = 1;
    }

    states[idx] = local_state;
}


int main(int argc, char *argv[]) {

    if (argc != 7) {
        std::cout << "Pass 6 arguments! Number of nodes, density, population, generations, mutation rate and block size!" << std::endl;
        return 1;
    }

    n = std::stoi(argv[1]);
    p = std::stod(argv[2]);
    max_population = std::stoi(argv[3]);
    max_generations = std::stoi(argv[4]);
    mutation_rate = std::stof(argv[5]);
    block_size = std::stof(argv[6]);

    graphName = "n" + std::to_string(n) + "p0" + std::to_string((int)(p * 10));
    

    std::string graphPath = "graphs/" + graphName + ".json";
    

    int* h_adj_matrix = new int[n * n];
    int* h_must_nodes;

    load_graph(graphPath, h_adj_matrix, n, h_must_nodes, must_nodes_count);
    

    int* h_population = new int[max_population * n];
    float* h_fitnesses = new float[max_population];

    
    int* d_adj_matrix;
    cudaMalloc(&d_adj_matrix, sizeof(int) * n * n);
    cudaMemcpy(d_adj_matrix, h_adj_matrix, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    int* d_population;
    cudaMalloc(&d_population, sizeof(int) * n * max_population);

    float* d_fitnesses;
    cudaMalloc(&d_fitnesses,sizeof(float) * max_population);

    int* d_new_population;
    cudaMalloc(&d_new_population, sizeof(int) * n * max_population);
    cudaMemcpy(d_population, h_population, max_population * n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_must_nodes;
    cudaMalloc(&d_must_nodes, sizeof(int) * must_nodes_count);
    cudaMemcpy(d_must_nodes, h_must_nodes, sizeof(int) * must_nodes_count, cudaMemcpyHostToDevice);
    

    curandState* d_states;
    cudaMalloc(&d_states, sizeof(curandState) * max_population);


    init_curand_states<<<(max_population + block_size - 1) / block_size, block_size>>>(d_states, time(NULL), max_population);


    clock_t begin = clock();

    int number_of_edges = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            if (h_adj_matrix[i *n + j] == 1) {
                number_of_edges++;
            }
        }
    }


    initialize_population_kernel<<<(max_population + block_size - 1)/block_size, block_size>>>(
        d_population, n, max_population, d_must_nodes, must_nodes_count, true, d_states
    );


    for (int g = 0; g < generations; g++) {
        fitness_kernel<<<(max_population + block_size - 1)/block_size, block_size>>>(
            d_population, d_fitnesses, d_adj_matrix, n, number_of_edges, max_population);

        reproduce_kernel<<<(max_population + block_size - 1) / block_size, block_size>>>(
            d_population, d_new_population, d_fitnesses, n, max_population, mutation_rate,
            d_must_nodes, must_nodes_count, d_states);

        std::swap(d_population, d_new_population);
    }


    fitness_kernel<<<(max_population + block_size - 1)/block_size, block_size>>>(
            d_population, d_fitnesses, d_adj_matrix, n, number_of_edges, max_population);


    cudaMemcpy(h_population, d_population, max_population * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitnesses, d_fitnesses, max_population * sizeof(float), cudaMemcpyDeviceToHost);


    int best_idx = 0;
    double best_fit = h_fitnesses[0];
    for (int fit = 0; fit < max_population; fit++) {
        if (h_fitnesses[fit] > best_fit) {
            best_fit = h_fitnesses[fit];
            best_idx = fit;
        }
    }


    int* result = new int[n];
    int out_size = 0;
    for (int i = 0; i < n; i++) {
        if (h_population[best_idx * n + i] == 1) {
            result[out_size++] = i;
        }
    }


    clock_t end = clock();
    int time = static_cast<int>(1e6 * (end - begin) / CLOCKS_PER_SEC);

    save_data(n, p, time, graphName, out_size, result);



    delete[] result;

    delete[] h_population;
    delete[] h_must_nodes;
    delete[] h_fitnesses;
    delete[] h_adj_matrix;


    cudaFree(d_population);
    cudaFree(d_new_population);
    cudaFree(d_fitnesses);
    cudaFree(d_adj_matrix);
    cudaFree(d_must_nodes);
    cudaFree(d_states);

    return 0;
}
