#include <iostream>
#include <sstream>
# include <arm_neon.h> 
#include <fstream>
#include <sys/time.h>
using namespace std;



unsigned int Ann[8399][264] = { 0 };
unsigned int Eli[8399][264] = { 0 };

const int Num = 263;
const int EliNum = 4535;
const int colNum = 8399;

void init_Eli()
{
    unsigned int a;
    ifstream infile("eli.txt");
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

void init_Ann()
{
    unsigned int a;
    ifstream infile("ann.txt");
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





void serial()
{
    uint32x4_t va_Eli =  vmovq_n_u32(0);
    uint32x4_t va_Ann =  vmovq_n_u32(0);
    bool sign;
    do
    {
        int i;
        for (i = colNum - 1; i - 8 >= -1; i -= 8)
        {
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
                                unsigned int temp = Eli[j][num];
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

        for (i = i + 8; i >= 0; i--)
        {
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
                                unsigned int temp = Eli[j][num];
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
        sign = false;
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
                sign = true;
            }
        }

    }while (sign == true);

}


int main()
{

    struct timeval starttime,endtime;
    double seconds;

    init_Ann();
    init_Eli();
    gettimeofday(&starttime, NULL);
    serial();
    gettimeofday(&endtime, NULL);
    seconds = ((endtime.tv_sec - starttime.tv_sec)*1000000 + (endtime.tv_usec - starttime.tv_usec)) / 1000.0;
    cout<<"use "<<seconds<<" ms"<<endl;


}
