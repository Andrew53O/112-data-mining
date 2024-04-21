#include<bits/stdc++.h>
using namespace std;
int main()
{
    ifstream in;
    string str;
    stringstream ss;
    char c;
    int n = 0;
    in.open("A/train_data.csv");
    in >> str;
    while(in >> str)
    {
        ss << str;
        n++;
    }
    int data[n][9];
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            ss >> data[i][j];
            if(ss >> c);
        }
    }

    return 0;
}
