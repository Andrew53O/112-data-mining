/*
 * Program Name: KNN
 * Completion Date: 2024-04-22
 * Author: Leo
 */
#include<bits/stdc++.h>
using namespace std;
// 懷孕次數、葡萄糖濃度、舒張壓、三頭肌皮皺厚度、胰島素濃度
float Pregnancies = 1, Glucose = 5, BloodPressure = 1, SkinThickness = 1, Insulin = 1;
// BMI、糖尿病函數、年齡
float BMI = 2, DiabetesPedigreeFunction = 10, Age = 1;

float Proportion[8] = {Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age};

int KNN_Range = 11;

string train_file = "B/train_data.csv", test_file = "B/test_data.csv";

vector<vector<float>> data;
ifstream in;
string str;
stringstream ss;
char c;
int n = 0, Correct = 0;

struct Distance{
    float value = 0;
    bool Outcome = 0;
};

bool cmp(Distance a, Distance b) {
    return a.value < b.value;
}

bool test(int N) {
    float input[9];
    int Outcome = 0;
    Distance d[n];
    for(int i = 0; i < n; i++) {
        d[i].Outcome = data[i][8];
    }
    ss << str << ',';
    for(int i = 0; i < 9; i++) {
        ss >> input[i];
        if(ss >> c);
    }
    for(int i = 0; i < n; i++) {
        float total = 0;
        for(int j = 0; j < 8; j++) {
            if(input[j] == 0 || data[i][j] == 0)
                continue;
            else {
                d[i].value += pow(input[j] - data[i][j], 2) * Proportion[j];
                Proportion[j];
                total += Proportion[j];
            }
        }
        d[i].value = sqrt(d[i].value/total);
    }
    sort(d, d+n, cmp);
    for(int i = 0; i < N; i++) {
        Outcome += d[i].Outcome;
    }
    if(Outcome > N/2) {
        Correct += (input[8] == 1);
        return 1;
    }
    else {
        Correct += (input[8] == 0);
        return 0;
    }
}

void KNN(int N) {
    int num = 0;
    in.open(test_file); // 測試資料
    in >> str; //temporary
    while(in >> str) {
        //cout << test(N) << endl;
        test(N);
        num++;
        //system("pause");
    }
    in.close();
    cout << "正確率: " << Correct*100.0/num << "%\n";
}

void train_data() {
    float input;
    in.open(train_file); // 訓練資料
    in >> str; //temporary
    while(in >> str) {
        ss << str << ',';
        n++;
    }
    in.close();
    for(int i = 0; i < n; i++) {
        vector<float> temp;
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
    KNN(KNN_Range);
    return 0;
}
