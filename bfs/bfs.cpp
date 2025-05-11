#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int total_edges = 0;
    
    #pragma omp parallel for reduction(+:total_edges)
    for (int i = 0; i < frontier->count; i++) {
        int node = frontier->vertices[i];
        total_edges += outgoing_size(g, node);
    }
    
    int* local_counts = (int*)malloc(sizeof(int) * omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); i++) {
        local_counts[i] = 0;
    }
    
    //////// Parallel for loop over frontier vertices
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier->count; i++) {
            int node = frontier->vertices[i];
            
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];
            
            int local_frontier_count = local_counts[thread_id];
            
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
                        #pragma omp critical
                        {
                            int index = new_frontier->count++;
                            new_frontier->vertices[index] = outgoing;
                        }
                    }
                }
            }
            
            local_counts[thread_id] = local_frontier_count;
        }
    }
    
    free(local_counts);
}

void bottom_up_step(
    Graph g,
    int* frontier,
    int frontier_count,
    int* new_frontier,
    int* new_frontier_count,
    int* distances,
    int depth)
{
    *new_frontier_count = 0;
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < g->num_nodes; i++) {
        if (distances[i] == NOT_VISITED_MARKER) {
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            
            for (const Vertex* v = start; v < end; v++) {
                int incoming = *v;
                
                if (distances[incoming] == depth) {
                    distances[i] = depth + 1;
                    
                    int index = __sync_fetch_and_add(new_frontier_count, 1);
                    new_frontier[index] = i;
                    break;
                }
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    int* distances = sol->distances;
    
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        distances[i] = NOT_VISITED_MARKER;
    }
    
    distances[ROOT_NODE_ID] = 0;
    
    int* frontier = (int*)malloc(sizeof(int) * graph->num_nodes);
    int* new_frontier = (int*)malloc(sizeof(int) * graph->num_nodes);
    
    frontier[0] = ROOT_NODE_ID;
    int frontier_count = 1;
    int depth = 0;
    
    while (frontier_count > 0) {
        #ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
        #endif
        
        int new_frontier_count = 0;
        
        bottom_up_step(graph, frontier, frontier_count, new_frontier, &new_frontier_count, distances, depth);
        
        #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier_count, end_time - start_time);
        #endif
        
        int* temp = frontier;
        frontier = new_frontier;
        new_frontier = temp;
        
        frontier_count = new_frontier_count;
        depth++;
    }
    
    free(frontier);
    free(new_frontier);
}

void bfs_hybrid(Graph graph, solution* sol)
{
    const float alpha = 14.0; 
    const float beta = 24.0;  
    
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    
    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    
    int* bu_frontier = (int*)malloc(sizeof(int) * graph->num_nodes);
    int* bu_new_frontier = (int*)malloc(sizeof(int) * graph->num_nodes);
    int bu_frontier_count = 0;
    
    bool using_top_down = true;
    int depth = 0;
    
    while (frontier->count > 0 || bu_frontier_count > 0) {
        #ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
        #endif
        
        int num_edges_to_check = 0;
        int num_vertices = graph->num_nodes;
        float frontier_size = 0;
        
        if (using_top_down) {
            frontier_size = (float)frontier->count;
            for (int i = 0; i < frontier->count; i++) {
                int node = frontier->vertices[i];
                num_edges_to_check += outgoing_size(graph, node);
            }
        } else {
            frontier_size = (float)bu_frontier_count;
            int unvisited = 0;
            #pragma omp parallel for reduction(+:unvisited)
            for (int i = 0; i < graph->num_nodes; i++) {
                if (sol->distances[i] == NOT_VISITED_MARKER) {
                    unvisited++;
                }
            }
            num_edges_to_check = graph->num_edges * (unvisited / (float)graph->num_nodes);
        }
        
        float frontier_fraction = frontier_size / num_vertices;
        
        if (using_top_down && frontier_fraction > 1.0 / alpha) {
            using_top_down = false;
            
            bu_frontier_count = frontier->count;
            #pragma omp parallel for
            for (int i = 0; i < frontier->count; i++) {
                bu_frontier[i] = frontier->vertices[i];
            }
            
            vertex_set_clear(frontier);
            vertex_set_clear(new_frontier);
        } 
        else if (!using_top_down && frontier_fraction < 1.0 / beta) {
            using_top_down = true;
            
            vertex_set_clear(frontier);
            #pragma omp parallel for
            for (int i = 0; i < bu_frontier_count; i++) {
                #pragma omp critical
                {
                    frontier->vertices[frontier->count++] = bu_frontier[i];
                }
            }
            
            bu_frontier_count = 0;
        }
        
        if (using_top_down) {
            vertex_set_clear(new_frontier);
            top_down_step(graph, frontier, new_frontier, sol->distances);
            
            vertex_set* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        } else {
            int new_count = 0;
            bottom_up_step(graph, bu_frontier, bu_frontier_count, bu_new_frontier, &new_count, sol->distances, depth);
            
            int* tmp = bu_frontier;
            bu_frontier = bu_new_frontier;
            bu_new_frontier = tmp;
            bu_frontier_count = new_count;
        }
        
        #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", 
               using_top_down ? frontier->count : bu_frontier_count, 
               end_time - start_time);
        #endif
        
        depth++;
    }
    
    free(bu_frontier);
    free(bu_new_frontier);
}
