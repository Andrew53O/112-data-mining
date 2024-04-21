#include<bits/stdc++.h>
using namespace std;
// 懷孕次數、葡萄糖濃度、舒張壓、三頭肌皮皺厚度、胰島素濃度
float Pregnancies = 0.5, Glucose = 0.1, BloodPressure = 0.1, SkinThickness = 0.1, Insulin = 0.1;
// BMI、糖尿病函數、年齡
float BMI = 0.5, DiabetesPedigreeFunction = 0.1, Age = 0.1;

float Proportion[8] = {Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age};

string train_file = "A/train_data.csv", test_file = "A/test_data.csv";

vector<vector<float>> data;
ifstream in;
string str;
stringstream ss;
char c;
int n = 0, Correct = 0;

struct Distance{
    float value;
    bool Outcome = 0;
};

bool cmp(Distance a, Distance b) {
    return a.value < b.value;
}

bool test(int N) {
    int input[9], Outcome = 0;
    Distance d[n];
    for(int i = 0; i < n; i++) {
        d[i].Outcome = data[i][8];
    }
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
                d[i].value += pow(input[j] - data[i][j], 2) * Proportion[j];
            }
        }
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
    in.open(test_file); // 測試資料
    in >> str; //temporary
    while(in >> str) {
        cout << test(N) << endl;
    }
    in.close();
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

void Correct_rate() {
    cout << "正確率: " << Correct*100.0/n << "%\n";
}

int main()
{
    train_data();
    KNN(3);
    Correct_rate();
    return 0;
}
