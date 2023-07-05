# include <arm_neon.h> 
# include <sys/time.h>
# include <iostream>
using namespace std;
const int n=1000;
float A[n][n];
float B[n][n];

float32x4_t va = vmovq_n_f32(0);
float32x4_t vx = vmovq_n_f32(0);
float32x4_t vaij = vmovq_n_f32(0);
float32x4_t vaik = vmovq_n_f32(0);
float32x4_t vakj = vmovq_n_f32(0);

void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			A[i][j] = 0;
		}
		A[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			A[i][j] = rand();
	}
	for (int k = 0; k < n; k++)
	{
		for (int i = k + 1; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				A[i][j] += A[k][j];
			}
		}
	}
}

void f_ordinary()//平凡算法
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

void f_ordinary_cache()//平凡+cache优化
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; 
        }
    }
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
                A[i][j] = A[i][j] - B[k][i] * A[k][j];
            }
        }
    }
}

void f_pro_2()//并行只二层循环
{
    for (int k = 0; k < n; k++)
	{
	    float32x4_t vt=vmovq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];
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


void f_pro_3()//并行只优化三层循环
{
    int j;
    for (int k = 0; k < n; k++)
	{
	    for (j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&A[i][j], vaij);
			}
			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
			A[i][k] = 0;
		}
	}
}

void f_pro()//并行算法
{
    for (int k = 0; k < n; k++)
	{
	    float32x4_t vt=vmovq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)//二层
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];

        }
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(A[i][k]);

			for (j = k + 1; j+4 <= n; j+=4)//三层
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
	}
}


void f_pro_cache()并行+cache优化
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; 
        }
    }
    for (int k = 0; k < n; k++)
	{
	    float32x4_t vt=vmovq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];

        }
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(B[k][i]);

			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
		}
	}
}

void f_pro_alignment()//并行+对齐
{
    for(int k = 0;k < n; k++)
    {
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        int j = k + 1;
        while((k * n + j) % 4 != 0)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];//对齐
            j++;
        }
        for(;j + 4 <= n; j += 4)
        {
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        for(;j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1;i < n; i++)
        {
            vaik = vmovq_n_f32(A[i][k]);
            int j = k + 1;
            while((i * n + j) % 4 != 0)//对齐
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
                j++;
            }
            for(;j + 4 <= n;j += 4){
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj,vaik);
                vaij = vsubq_f32(vaij,vx);
                vst1q_f32(&A[i][j],vaij);
            }
            for(;j < n; j++){
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
    }
}

void getResult()
{
    for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << A[i][j] << " ";
		}
		cout << endl;
	}
}



int main()
{

    struct timeval head,tail;
    init();
    gettimeofday(&head, NULL);
    f_ordinary();
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"平凡算法: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_ordinary_cache();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"平凡+cache: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_pro_division();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行只优化二层循环: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_pro_elimination();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行只优化三层循环: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_pro();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_pro_cache();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行+cache: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);
    f_pro_alignment();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行+对齐: "<<seconds<<" ms"<<endl;

}
