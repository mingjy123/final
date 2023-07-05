#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <immintrin.h> 
#include <sys/time.h>
using namespace std;

unsigned int x[37960][1188] = { 0 };
unsigned int b[37960][1188] = { 0 };

const int Num = 1187;
const int pasNum = 14921;
const int lieNum = 37960;

void init_x()
{
    unsigned int a;
    ifstream infile("a2.txt");
    char fin[10000] = { 0 };
    int id;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                mark = 1;
                id = a;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            x[id][Num - 1 - j] += temp;
            x[id][Num] = 1;
        }
    }
}
void init_b()
{
    unsigned int a;
    ifstream infile("p2.txt");
    char fin[10000] = { 0 };
    int id = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                b[id][Num] = a;
                mark = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            b[id][Num - 1 - j] += temp;
        }
        id++;
    }
}
//平凡算法
void f_ordinary()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] <= i && b[j][Num] >= i - 7)
            {
                int index = b[j][Num];
                if (x[index][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^x[index][k];
                    }

                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[index][k] = b[j][k];
                    x[index][Num] = 1;
                    break;
                }

            }
        }
    }

    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] == i)
            {
                if (x[i][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[i][k] = b[j][k];

                    x[i][Num] = 1;
                    break;
                }
            }
        }
    }
}



__m128 va_Pas;
__m128 va_Act;
void f_sse()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] <= i && b[j][Num] >= i - 7)
            {
                int index = b[j][Num];
                if (x[index][Num] == 1)//并行
                {
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        va_Pas = _mm_loadu_ps((float*)&(b[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(x[index][k]));
                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(b[j][k]), va_Pas);
                    }

                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[index][k];
                    }

                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;
                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[index][k] = b[j][k];
                    x[index][Num] = 1;
                    break;
                }
            }
        }
    }

    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] == i)
            {
                if (x[i][Num] == 1)//并行
                {
                    int k;
                    for (k = 0; k + 4 <= Num; k += 4)
                    {
                        va_Pas = _mm_loadu_ps((float*)&(b[j][k]));
                        va_Act = _mm_loadu_ps((float*)&(x[i][k]));
                        va_Pas = _mm_xor_ps(va_Pas, va_Act);
                        _mm_store_ss((float*)&(b[j][k]), va_Pas);
                    }
                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[i][k] = b[j][k];

                    x[i][Num] = 1;
                    break;
                }
            }
        }
    }

}
__m256 va_Pas2;
__m256 va_Act2;

void f_avx256()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] <= i && b[j][Num] >= i - 7)
            {
                int index = b[j][Num];
                if (x[index][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        va_Pas2 = _mm256_loadu_ps((float*)&(b[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(x[index][k]));
                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(b[j][k]), va_Pas2);
                    }
                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[index][k] =b[j][k];

                    x[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] == i)
            {
                if (x[i][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 8 <= Num; k += 8)
                    {
                        va_Pas2 = _mm256_loadu_ps((float*)&(b[j][k]));
                        va_Act2 = _mm256_loadu_ps((float*)&(x[i][k]));
                        va_Pas2 = _mm256_xor_ps(va_Pas2, va_Act2);
                        _mm256_storeu_ps((float*)&(b[j][k]), va_Pas2);
                    }
                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^x[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;
                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[i][k] = b[j][k];

                    x[i][Num] = 1;
                    break;
                }
            }
        }
    }
}


__m512 va_Pas3;
__m512 va_Act3;

void f_avx512()
{
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] <= i && b[j][Num] >= i - 7)
            {
                int index = b[j][Num];
                if (x[index][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        va_Pas3 = _mm512_loadu_ps((float*)&(b[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(x[index][k]));
                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(b[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[index][k] = b[j][k];

                    x[index][Num] = 1;
                    break;
                }
            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < pasNum; j++)
        {
            while (b[j][Num] == i)
            {
                if (x[i][Num] == 1)
                {
                    int k;
                    for (k = 0; k + 16 <= Num; k += 16)
                    {
                        va_Pas3 = _mm512_loadu_ps((float*)&(b[j][k]));
                        va_Act3 = _mm512_loadu_ps((float*)&(x[i][k]));
                        va_Pas3 = _mm512_xor_ps(va_Pas3, va_Act3);
                        _mm512_storeu_ps((float*)&(b[j][k]), va_Pas3);
                    }

                    for (; k < Num; k++)
                    {
                        b[j][k] = b[j][k] ^ x[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (b[j][num] != 0)
                        {
                            unsigned int temp = b[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    b[j][Num] = S_num - 1;
                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        x[i][k] = b[j][k];

                    x[i][Num] = 1;
                    break;
                }
            }
        }
    }
}
int main()
{
    struct timeval head,tail;
    double seconds;
    init_x();
    init_c();
    gettimeofday(&head, NULL);
    f_ordinary();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"平凡算法: "<<seconds<<" ms"<<endl;

    init_x();
    init_c();
    gettimeofday(&head, NULL);
    f_sse();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"sse: "<<seconds<<" ms"<<endl;

    init_x();
    init_c();
    gettimeofday(&head, NULL);
    f_avx256();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"avx256: "<<seconds<<" ms"<<endl;

    init_x();
    init_c();
    gettimeofday(&head, NULL);
    f_avx512();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"avx512: "<<seconds<<" ms"<<endl;


}
