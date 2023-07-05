#include <omp.h>
#include <iostream>
#include <semaphore.h>
#include <sstream>
# include <arm_neon.h> 
#include <fstream>
#include <sys/time.h>
using namespace std;



unnexted int Ann[8399][264] = { 0 };
unnexted int Eli[8399][264] = { 0 };

const int Num = 263;
const int EliNum = 4535;
const int colNum = 8399;
const int NUM_THREADS = 7; 

bool next;
struct threadParam_t
{
    int t_id; 
};

void init_Ann()
{
    unnexted int a;
    ifstream infile("Ann.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                index = a;
                mark = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Ann[index][Num - 1 - j] += temp;
            Ann[index][Num] = 1;
        }
    }
}

void init_Eli()
{
    unnexted int a;
    ifstream infile("Eli.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                Eli[index][Num] = a;
                mark = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Eli[index][Num - 1 - j] += temp;
        }
        index++;
    }
}

void gauss_omp()
{
    uint32x4_t va_Eli =  vmovq_n_u32(0);
    uint32x4_t va_Ann =  vmovq_n_u32(0);
    bool next;
    #pragma omp parallel num_threads(NUM_THREADS), private(va_Eli, va_Ann)
    do
    {

        for (int i = colNum - 1; i - 8 >= -1; i -= 8)
        {
            #pragma omp for schedule(static)
            for (int j = 0; j < EliNum; j++)
            {
                while (Eli[j][Num] <= i && Eli[j][Num] >= i - 7)
                {
                    int index = Eli[j][Num];

                    if (Ann[index][Num] == 1)
                    {
                        int k;
                        for (k = 0; k+4 <= Num; k+=4)
                        {
                            va_Eli =  vld1q_u32(& (Eli[j][k]));
                            va_Ann =  vld1q_u32(& (Ann[index][k]));

                            va_Eli = veorq_u32(va_Eli,va_Ann);
                            vst1q_u32( &(Eli[j][k]) , va_Eli );
                        }

                        for( ; k<Num; k++ )
                        {
                            Eli[j][k] = Eli[j][k] ^ Ann[index][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Eli[j][num] != 0)
                            {
                                unnexted int temp = Eli[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Eli[j][Num] = S_num - 1;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        for (int i = colNum%8-1; i >= 0; i--)
        {
            #pragma omp for schedule(static)
            for (int j = 0; j < EliNum; j++)
            {
                while (Eli[j][Num] == i)
                {
                    if (Ann[i][Num] == 1)
                    {
                        int k;
                        for (k = 0; k+4 <= Num; k+=4)
                        {
                            va_Eli =  vld1q_u32(& (Eli[j][k]));
                            va_Ann =  vld1q_u32(& (Ann[i][k]));

                            va_Eli = veorq_u32(va_Eli,va_Ann);
                            vst1q_u32( &(Eli[j][k]) , va_Eli );
                        }

                        for( ; k<Num; k++ )
                        {
                            Eli[j][k] = Eli[j][k] ^ Ann[i][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++)
                        {
                            if (Eli[j][num] != 0)
                            {
                                unnexted int temp = Eli[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Eli[j][Num] = S_num - 1;

                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

    #pragma omp single
    {
        next = false;
        for (int i = 0; i < EliNum; i++)
        {
            int temp = Eli[i][Num];
            if (temp == -1)
            {
                continue;
            }

            if (Ann[temp][Num] == 0)
            {
                for (int k = 0; k < Num; k++)
                    Ann[temp][k] = Eli[i][k];
                Eli[i][Num] = -1;
                next = true;
            }
        }
    }

    }while (next == true);

}
void serial()
{
    int i;
    for (i = colNum-1; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < EliNum; j++)
        {
            while (Eli[j][Num] <= i && Eli[j][Num] >= i - 7)
            {
                int index = Eli[j][Num];
                if (Ann[index][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        Eli[j][k] = Eli[j][k] ^ Ann[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Eli[j][num] != 0)
                        {
                            unnexted int temp = Eli[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Eli[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Ann[index][k] = Eli[j][k];

                    Ann[index][Num] = 1;
                    break;
                }

            }
        }
    }
    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < EliNum; j++)
        {
            while (Eli[j][Num] == i)
            {
                if (Ann[i][Num] == 1)
                {
                    for (int k = 0; k < Num; k++)
                    {
                        Eli[j][k] = Eli[j][k] ^ Ann[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++)
                    {
                        if (Eli[j][num] != 0)
                        {
                            unnexted int temp = Eli[j][num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Eli[j][Num] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < Num; k++)
                        Ann[i][k] = Eli[j][k];

                    Ann[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

int main()
{
    init_Ann();
    init_Eli();
    struct timeval starttime,endtime;
    double seconds;

    gettimeofday(&starttime, NULL);
    gauss_omp();
    gettimeofday(&endtime, NULL);
    seconds = ((endtime.tv_sec - starttime.tv_sec)*1000000 + (endtime.tv_usec - starttime.tv_usec)) / 1000.0;
    cout<<"use "<<seconds<<" ms"<<endl;


}