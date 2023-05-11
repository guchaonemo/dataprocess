#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<string>
#include <sstream>
#include <cstring>
#include<numeric>
#include<omp.h>
#include<chrono>
#include<immintrin.h>
#define USE_AVX256
#define USE_AVX512
double sum_of_square_avx256(double* a, double* b, int n);
double sum_of_square_avx512(double* a, double* b, int n);
std::vector<std::vector<double>> load_txt(const char* filename);
void compute_distance(std::vector<std::vector<double>> &distance, std::vector<std::vector<double>> &data);
void generate_index(std::vector<std::vector<double>> &data, std::vector<std::vector<int>> &index);
void initial_index(std::vector<std::vector<int>> &index,int n);
double sum_of_square(double* a, double* b,int n);
template<typename T>
void tag_sort(const std::vector<T>& v, std::vector<int>& result);
void transpose_index(std::vector<std::vector<int>> &index, std::vector<std::vector<int>> &index_t);
void compute_RNN_KNN_RN(std::vector<std::vector<int>>& index, std::vector<std::vector<int>> &RNN, std::vector<std::vector<int>>& KNN, std::vector<int > &RN);
std::vector<int> intersection(std::vector<int> &v1, std::vector<int> &v2);
void compute_NaN(std::vector<std::vector<int>>& NaN, std::vector<std::vector<int>> &RNN, std::vector<std::vector<int>>& KNN);
void compute_NaN_Con(std::vector<std::vector<int>> &NaN_con, std::vector<std::vector<int>> &NaN,
	std::vector<std::vector<int>> &KNN, std::vector<std::vector<bool>> &NaN_out,int &nonzero);

int main()
{
	const char* filename = "Optdigits_test_data.txt";
	std::vector<std::vector<int>> index;
	int n;

	std::vector<std::vector<double>> data = load_txt(filename);
	n = data.size();

	double total_time = 0.0;
	auto start_t = std::chrono::steady_clock::now();
	initial_index(index,  n);

	std::vector<std::vector<double>> distance(n, std::vector<double>(n, 0.0));
	compute_distance(distance, data);

	generate_index(distance, index);

	std::vector<std::vector<int>> index_t(n, std::vector<int>(n));

	transpose_index(index, index_t);


	std::vector<std::vector<int>> RNN(n, std::vector<int>());
	std::vector<std::vector<int>> KNN(n, std::vector<int>());
	std::vector<std::vector<int>> NaN(n, std::vector<int>());

	std::vector<std::vector<int>> NaN_con(n, std::vector<int>(n,0));
	std::vector<std::vector<bool>> NaN_out(n, std::vector<bool>(n, 0));
	std::vector<int > RN(n, 0);
	int nonzero = 0;

	compute_RNN_KNN_RN(index_t, RNN, KNN, RN);

	compute_NaN(NaN, RNN, KNN);

	compute_NaN_Con(NaN_con, NaN, KNN, NaN_out, nonzero);
	auto end_t = std::chrono::steady_clock::now();
	total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count();
	printf("compute NaN_con %.4lf ms \n", total_time/1000000);
	FILE* fp = fopen("data.txt", "w");
	fprintf(fp, "%d\n", n);
	fprintf(fp, "%d\n", nonzero);
	for(int j=0;j<n;j++)
		for (int i = 0; i < n; i++)
		{
			if (NaN_out[i][j])
			{
				fprintf(fp, "%d %d\n", j , i);
			}
		}
	fclose(fp);
	return 0;
}
std::vector<std::vector<double>> load_txt(const char* filename)
{
	std::vector<std::vector<double>> ret;
	std::ifstream readin(filename);
	std::string line;
	int nlines = 0;
	double local;
	while (std::getline(readin, line))
	{
		nlines++;
		std::istringstream iss(line);
		int counts = 0;
		std::vector<double> local_vec;
		while (iss >> local)
		{
			local_vec.push_back(local);
			counts++;
		}
		ret.push_back(local_vec);

	}
	readin.close();
	return ret;
}
void initial_index(std::vector<std::vector<int>> &index,int n)
{
	index.resize(n, std::vector<int>(n, 0));

	std::iota(index[0].begin(), index[0].end(), 1);
	//for (int i = 0; i < n; i++)
	//{
	//	index[0][i] = i + 1;
	//}
	for (int i = 1; i < n; i++)
	{
		memcpy(index[i].data(), index[0].data(), sizeof(int)*n);
		//if (index[i][n - 1] != n)
		//	printf("memcpy error \n");
	}
}
void generate_index(std::vector<std::vector<double>> &data, std::vector<std::vector<int>> &index)
{
	int n = data.size();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		tag_sort(data[i], index[i]);
	}
}
void compute_distance(std::vector<std::vector<double>> &distance, std::vector<std::vector<double>> &data)
{
	int n = distance.size();
	int L = data[0].size();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		double* a = data[i].data();
		for (int j = i; j < n; j++)
		{
			distance[i][j] = sum_of_square_avx256(a, data[j].data(), L);
		}
	}

#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j <i; j++)
		{
			distance[i][j] = distance[j][i];
		}
	}
}
double sum_of_square(double* a, double* b, int n)
{
	double ret = 0;
	for (int i = 0; i < n; i++)
		ret += (a[i] - b[i])*(a[i] - b[i]);
	return ret;
}
double sum_of_square_avx512(double* a, double* b, int n)
{
	double result = 0.0;
	int i = 0;
#ifdef USE_AVX512
	__m512d ret = _mm512_set1_pd(0.0);
	int loops = n / 8;
	for (; i < loops; i++)
	{
		__m512d la = _mm512_loadu_pd(a + 8*i);
		__m512d lb = _mm512_loadu_pd(b + 8*i);
		lb = _mm512_sub_pd(la, lb);
		//a*b+c
		ret = _mm512_fmadd_pd(lb, lb, ret);
	}
	double rets[8];
	_mm512_storeu_pd(rets, ret);
	result = rets[0] + rets[1] + rets[2] + rets[3]
		   + rets[4] + rets[5] + rets[6] + rets[7];
	i = 8 * loops;
#endif
	for (; i < n; i++)
		result += (a[i] - b[i])*(a[i] - b[i]);
	return result;
}
double sum_of_square_avx256(double* a, double* b, int n)
{
	double result = 0.0;
	int i = 0;
#ifdef USE_AVX256
	__m256d ret = _mm256_set1_pd(0.0);
	int loops = n / 4;
	for (; i < loops; i++)
	{
		__m256d la = _mm256_loadu_pd(a + 4*i);
		__m256d lb = _mm256_loadu_pd(b + 4*i);
		lb = _mm256_sub_pd(la, lb);
		ret = _mm256_fmadd_pd(lb, lb, ret);
	}
	double rets[4];
	_mm256_storeu_pd(rets, ret);
	result = rets[0] + rets[1] + rets[2] + rets[3];
	i = 4 * loops;
#endif
	for (; i < n; i++)
		result += (a[i] - b[i])*(a[i] - b[i]);
	return result;
}

template<typename T>
void tag_sort(const std::vector<T>& v, std::vector<int>& result)
{
	int n = result.size();
	std::vector<std::pair<T, int>> value_idx(n);
	for (int i = 0; i < n; i++)
	{
		value_idx[i].first = v[i];
		value_idx[i].second = result[i];
	}
	std::sort(value_idx.begin(), value_idx.end());
	for (int i = 0; i < n; i++)
	{
		result[i]=value_idx[i].second ;
	}
}

void transpose_index(std::vector<std::vector<int>> &index, std::vector<std::vector<int>> &index_t)
{
	int n = index.size();

#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			index_t[i][j] = index[j][i];
	}
}

void compute_RNN_KNN_RN(std::vector<std::vector<int>>& index,std::vector<std::vector<int>> &RNN, std::vector<std::vector<int>>& KNN, std::vector<int > &RN)
{
	int r = 1;
	int tag = 1;
	int n = index.size();
	std::vector<int> cnt(n, 0);
	while (tag)
	{
		std::vector<int>& KNN_idx = index[r];
		for (int i = 0; i < n; i++)
		{
			RNN[KNN_idx[i] - 1].push_back(i + 1);
			KNN[i].push_back(KNN_idx[i]);
		}
		


		int counts = 0;
		for (int i = 0; i < n; i++)
		{
			if (RNN[i].size() > 0)
			{
				RN[i] = 1;
				counts++;
			}
		}
		for (int i = 0; i < n; i++)
		{
			if (RN[i] == 0)
				cnt[r - 1]++;
		}
		if (r > 2 && cnt[r - 1] == cnt[r - 2])
		{
			tag = 0;
			r = r - 1;
		}
		r = r + 1;
		//printf("this is ok counts=%d\n", counts);
	}
}

std::vector<int> intersection(std::vector<int> &v1, std::vector<int> &v2) 
{
	std::vector<int> v3;

	std::sort(v1.begin(), v1.end());
	std::sort(v2.begin(), v2.end());

	std::set_intersection(v1.begin(), v1.end(),
		v2.begin(), v2.end(),
		back_inserter(v3));
	return v3;
}

void compute_NaN(std::vector<std::vector<int>>& NaN, std::vector<std::vector<int>> &RNN, std::vector<std::vector<int>>& KNN)
{
	int n = RNN.size();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		NaN[i] = intersection(RNN[i], KNN[i]);
	}
}

void compute_NaN_Con(std::vector<std::vector<int>> &NaN_con, std::vector<std::vector<int>> &NaN,
	std::vector<std::vector<int>> &KNN, std::vector<std::vector<bool>> &NaN_out,int &nonzero)
{
	int n = NaN_con.size();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int n_length = NaN[i].size();
		if (n_length == 0)
		{
			NaN_con[i][KNN[i][0] - 1] = 1;
		}
		else
		{
			for (int j = 0; j < n_length; j++)
				NaN_con[i][NaN[i][j] - 1] = 1;
		}
	}
	std::vector<int> ncounts(n, 0);
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
		{
			NaN_out[i][j] = ((NaN_con[i][j] + NaN_con[j][i]) > 0);
			if (NaN_out[i][j])ncounts[i]++;
		}
	for (int i = 0; i < n; i++)
		nonzero += ncounts[i];
}