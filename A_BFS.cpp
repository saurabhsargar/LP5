#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

int main() {
    int V, E, u, v, start;
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;

    vector<vector<int>> adj(V);
    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; ++i) {
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    cout << "Enter starting node: ";
    cin >> start;

    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    cout << "BFS Traversal: ";
    while (!q.empty()) {
        int size = q.size();
        vector<int> next;

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            int node;
            #pragma omp critical
            {
                node = q.front(); q.pop();
                cout << node << " ";
            }

            for (int nei : adj[node]) {
                if (!visited[nei]) {
                    #pragma omp critical
                    {
                        if (!visited[nei]) {
                            visited[nei] = true;
                            next.push_back(nei);
                        }
                    }
                }
            }
        }

        for (int n : next) q.push(n);
    }
    cout << endl;
    return 0;
}
