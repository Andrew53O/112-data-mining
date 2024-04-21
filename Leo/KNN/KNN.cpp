#include<bits/stdc++.h>
using namespace std;
// 懷孕次數、葡萄糖濃度、舒張壓、三頭肌皮皺厚度、胰島素濃度
float Pregnancies = 0.1, Glucose = 0.1, BloodPressure = 0.1, SkinThickness = 0.1, Insulin = 0.1;
// BMI、糖尿病函數、年齡
float BMI = 0.1, DiabetesPedigreeFunction = 0.1, Age = 0.1;

float Proportion[8] = {Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age};

string train_file = "A/train_data.csv", test_file = "A/test_data.csv";

vector<vector<float>> data;
ifstream in;
string str;
stringstream ss;
char c;
int n = 0;

struct distance{
    int value;
    bool Outcome = 0;
};

bool KNN() {
    in.open(test_file); // 測試資料
    in >> str; //temporary
    while(in >> str) {
        int input[9];
        ss << str;
        for(int i = 0; i < 9; i++) {
            ss >> input[i];
            if(ss >> c);
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < 8; j++) {
                if(input[j] == 0 || data[i][j] == 0)
                    continue;
                else {

                }
            }
        }
    }
}

void train_data() {
    float input;
    in.open(train_file); // 訓練資料
    in >> str; //temporary
    while(in >> str) {
        ss << str;
        n++;
    }
    in.close();
    for(int i = 0; i < n; i++) {
        vector<float> temp(8);
        for(int j = 0; j < 9; j++) {
            ss >> input;
            temp.push_back(input);
            if(ss >> c);
        }
        data.push_back(temp);
    }
}

int main()
{
    train_data();
    KNN();
    return 0;
}
