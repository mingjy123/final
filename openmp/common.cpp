#include <omp.h>
#include <iostream>
#include <windows.h>
using namespace std;

const int n = 1000;
float arr[n][n];
float A[n][n];
const int NUM_THREADS = 7; 


void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = 0;
		}
		arr[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			arr[i][j] = rand() % 100;
	}

	for (int i = 0; i < n; i++)
	{
		int k1 = rand() % n;
		int k2 = rand() % n;
		for (int j = 0; j < n; j++)
		{
			arr[i][j] += arr[0][j];
			arr[k1][j] += arr[k2][j];
		}
	}
}


void anotherTest()
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            A[i][j]=arr[i][j];
    }
}


void serial()
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

void omp_static()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
	}
}


void omp_dynamic()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		#pragma omp for schedule(dynamic, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
	}
}

void omp_guided()
{
	 #pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < n; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}
		#pragma omp for schedule(guided, 80)
		for (int i = k + 1; i < n; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
	}
}


int main()
{
	init();
    double seconds ;
    long long head,tail,freq,noww;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);



	anotherTest();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	serial();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    seconds = (tail - head) * 1000.0 / freq ;
	cout << "serial: " << seconds << " ms" << endl;



	anotherTest();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_static();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
	seconds = (tail - head) * 1000.0 / freq ;
	cout << "omp_static: " << seconds << " ms" << endl;



	anotherTest();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_dynamic();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
	seconds = (tail - head) * 1000.0 / freq ;
	cout << "omp_dynamic: " << seconds << " ms" << endl;


	anotherTest();
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	omp_guided();
	QueryPerformanceCounter((LARGE_INTEGER *)&tail );
	seconds = (tail - head) * 1000.0 / freq ;
	cout << "omp_guided: " << seconds << " ms" << endl;

}

