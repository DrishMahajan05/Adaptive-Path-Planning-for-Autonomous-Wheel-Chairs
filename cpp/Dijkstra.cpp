#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// Structure to represent an edge
struct Edge {
    int v;
    int wt;
};

void dijkstra(int src, vector<vector<Edge>> g, int V){
    vector<int> dist(V,INT_MAX);
    dist[src]=0;
    priority_queue<pair<int,int>,vector<pair<int,int>>, greater<pair<int,int>>> pq;
    pq.push({0, src});
    while(pq.size()>0){
        int u=pq.top().second;
        pq.pop();

        for(Edge e: g[u]){
            if(dist[e.v]>dist[u]+e.wt){
                dist[e.v]=dist[u]+e.wt;
                pq.push({dist[e.v],e.v});
            }
        }
    }
    for(int i=0;i<V;i++){
        cout<<dist[i]<<" ";
    }
    cout<<endl;
}


int main() {
    int V = 6;
    // Correctly nested vector representation for an adjacency list
    vector<vector<Edge>> graph(V);

    // Adding edges to the graph
    graph[0].push_back({1, 2});
    graph[0].push_back({2, 4});
    graph[1].push_back({2, 1});
    graph[1].push_back({3, 7});
    graph[2].push_back({4, 3});
    graph[3].push_back({5, 1});
    graph[4].push_back({3, 2});
    graph[4].push_back({5, 5});

    dijkstra(0, graph, V);

    return 0;
}