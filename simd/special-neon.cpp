# include <arm_neon.h> 
# include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;
unsigned int x[37960][1188] = { 0 };
unsigned int b[37960][1188] = { 0 };

void init_x()
{
    unsigned int a;
    ifstream infile("act.txt");
    char fin[100000] = { 0 };
    int id;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                id = a;
                mark = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            x[id][1186 - j] += temp;
            x[id][1187] = 1;
        }
    }
}
void init_b()
{
    unsigned int a;
    ifstream infile("pas.txt");
    char fin[100000] = { 0 };
    int id = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int mark = 0;
        while (line >> a)
        {
            if (mark == 0)
            {
                b[id][1187] = a;
                mark = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            b[id][1186 - j] += temp;
        }
        id++;
    }
}


void f_ordinary()
{
    int i;
    for (i = 37959; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (b[j][1187] <= i && b[j][1187] >= i - 7)
            {
                int index = b[j][1187];
                if (x[index][1187] == 1)
                {
                    for (int k = 0; k < 1187; k++)
                    {
                        b[j][k] = b[j][k] ^ x[index][k];
                    }

                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    b[j][1187] = S_num - 1;

                }
                else//消元子为空
                {
                    for (int k = 0; k < 1187; k++)
                        x[index][k] = b[j][k];
                    x[index][1187] = 1;
                    break;
                }

            }
        }
    }


    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (b[j][1187] == i)
            {
                if (x[i][1187] == 1)
                {
                    for (int k = 0; k < 1187; k++)
                    {
                        b[j][k] = b[j][k] ^ x[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    b[j][1187] = S_num - 1;

                }
                else//消元子为空
                {
                    for (int k = 0; k < 1187; k++)
                        x[i][k] = b[j][k];
                    x[i][1187] = 1;
                    break;
                }
            }
        }
    }
}

void f_pro()
{
    int i;
    for (i = 37959; i - 8 >= -1; i -= 8)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (b[j][1187] <= i && b[j][1187] >= i - 7)
            {
                int index = b[j][1187];
                if (x[index][1187] == 1)//并行优化
                {
                    int k;
                    for (k = 0; k+4 <= 1187; k+=4)
                    {
                        uint32x4_t vaPas =  vld1q_u32(& (b[j][k]));
                        uint32x4_t vaAct =  vld1q_u32(& (x[index][k]));
                        vaPas = veorq_u32(vaPas,vaAct);
                        vst1q_u32( &(b[j][k]) , vaPas );
                    }

                    for( ; k<1187; k++ )
                    {
                        b[j][k] = b[j][k] ^x[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    b[j][1187] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < 1187; k++)
                        x[index][k] = b[j][k];
                    x[index][1187] = 1;
                    break;
                }
            }
        }
    }
    for (i = i + 8; i >= 0; i--)
    {
        for (int j = 0; j < 14921; j++)
        {
            while (b[j][1187] == i)
            {
                if (x[i][1187] == 1)//并行优化
                {

                    int k;
                    for (k = 0; k+4 <= 1187; k+=4)
                    {
                        uint32x4_t va_Pas =  vld1q_u32(& (b[j][k]));
                        uint32x4_t va_Act =  vld1q_u32(& (x[i][k]));

                        va_Pas = veorq_u32(va_Pas,va_Act);
                        vst1q_u32( &(b[j][k]) , va_Pas );
                    }

                    for( ; k<1187; k++ )
                    {
                        b[j][k] = b[j][k] ^ x[i][k];
                    }

                    int num = 0, S_num = 0;
                    for (num = 0; num < 1187; num++)
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
                    b[j][1187] = S_num - 1;

                }
                else
                {
                    for (int k = 0; k < 1187; k++)
                        x[i][k] = b[j][k];

                    x[i][1187] = 1;
                    break;
                }
            }
        }
    }
}


int main()
{

    struct timeval head,tail;

    init_x();
    init_b();
    gettimeofday(&head, NULL);
    f_ordinary();
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"平凡算法: "<<seconds<<" ms"<<endl;

    init_x();
    init_b();
    gettimeofday(&head, NULL);
    f_pro();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout<<"并行优化: "<<seconds<<" ms"<<endl;


}
