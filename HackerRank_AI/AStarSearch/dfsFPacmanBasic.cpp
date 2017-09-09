#include <iostream>
#include <vector>
#include <array>
#include <stack>
using namespace std;

bool isAllowed(int i, int j, int m, int n, vector<string> grid, vector<vector<int>> visited) {
    return (i>=0 && j>=0 && i<m && j<n && grid[i][j]!='%' && visited[i][j]==0);
}

void dfs( int r, int c, int pacman_r, int pacman_c, int food_r, int food_c, vector <string> grid){
    array<array<int,2>,4> dir = {{{-1,0}, {0,-1}, {0,1}, {1,0}}};   // UP --> LEFT --> RIGHT --> DOWN
    stack<pair<int,int>> dfsStack;
    dfsStack.emplace(make_pair(pacman_r, pacman_c));    // start first location
    vector<vector<int>> visited(r, vector<int>(c, 0));  // visited vector for dfs
    vector<pair<int,int>> visitingOrder;
    // print all the nodes when you visiting
    while(!dfsStack.empty()){
        pair<int,int> t = dfsStack.top();
        dfsStack.pop();
        visitingOrder.emplace_back(t);                     // store the visiting order 
        visited[t.first][t.second] = 1;                    // mark location as visited
        if (t.first==food_r && t.second==food_c) {
            cout << visitingOrder.size() << endl;
            for (pair<int,int> f : visitingOrder) {
                cout << f.first << ' ' << f.second << endl;
            }
            break;
        }
        
        for (int d=0; d<dir.size(); ++d) {
            int x = dir[d][0];  
            int y = dir[d][1];
            if (isAllowed(x,y,r,c,grid,visited)){
                dfsStack.emplace(make_pair(x,y));
            }
        }
    }   
}

int main(void) {

    int r,c, pacman_r, pacman_c, food_r, food_c;
    
    cin >> pacman_r >> pacman_c;
    cin >> food_r >> food_c;
    cin >> r >> c;
    
    vector <string> grid;

    for(int i=0; i<r; i++) {
        string s; cin >> s;
        grid.push_back(s);
    }

    dfs( r, c, pacman_r, pacman_c, food_r, food_c, grid);

    return 0;
}

