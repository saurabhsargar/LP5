#include <iostream>
#include <vector>
#include <stack>
#include <algorithm> // For sort
using namespace std;

int main() {
    int V, E, u, v, start;
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;

    vector<vector<int>> adj(V);
    vector<bool> visited(V, false);

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; ++i) {
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected
    }

    // Sort adjacency list to ensure left-to-right DFS order
    for (int i = 0; i < V; ++i)
        sort(adj[i].begin(), adj[i].end());

    cout << "Enter starting node: ";
    cin >> start;

    stack<int> s;
    s.push(start);

    cout << "DFS Traversal: ";
    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";

            // Push neighbors in reverse order so smallest is processed first
            for (int i = adj[node].size() - 1; i >= 0; --i) {
                int neighbor = adj[node][i];
                if (!visited[neighbor])
                    s.push(neighbor);
            }
        }
    }

    cout << endl;
    return 0;
}
