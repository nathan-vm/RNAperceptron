//MLP DECIFRAR

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>

#define ENTRADAS        5
#define SAIDAS          1
#define NR_AMOSTRAS     26
#define NR_NEURON_O     12
#define EPOCAS          10000
#define TX_APRENDIZADO  1
#define MOMENTUM        0.1
#define MAX_IN          1.0
#define MAX_OUT         1000.0
#define ZERO_TEST       0


/*--------------------------------------------------
   Dados para o treinamento da rede
 -------------------------------------------------*/
int cj_treinamento[NR_AMOSTRAS][ENTRADAS+SAIDAS] = {
        {0,0,0,0,0,65}, //a
        {0,0,0,0,1,66}, //b
        {0,0,0,1,0,67}, // c ...
        {0,0,1,0,0,68}, //d
        {0,1,0,0,0,69}, //e
        {1,0,0,0,0,70},//f
        {1,1,0,0,0,71},//g
        {0,1,1,0,0,72},//h
        {0,0,1,1,0,73},//i
        {0,0,0,1,1,74},//j
        {0,0,1,1,1,75},//k
        {0,1,1,1,0,76},//l
        {1,1,1,0,0,77},//m
        {1,1,1,1,0,78},//n
        {0,1,1,1,1,79},
        {1,1,1,1,1,80},
        {1,0,0,0,1,81},
        {0,1,0,1,0,82},
        {1,1,0,0,1,83},
        {0,1,1,0,1,84},
        {1,0,0,1,0,85},
        {1,1,0,1,0,86},
        {1,1,0,1,1,87},
        {0,1,0,0,1,88},
        {0,0,1,0,1,89},
        {1,0,1,0,0,90} //z
};



/*
   Vari�veis globais
 */
double w_e_o[ENTRADAS+1][NR_NEURON_O];
double w_o_s[NR_NEURON_O+1][SAIDAS];
double saida_o[NR_NEURON_O];
double saida_s[SAIDAS];
double delta_saida[SAIDAS];
double gradiente_oculta[NR_NEURON_O];
double delta_oculta[NR_NEURON_O];
double media_erro = 0.0;

char frase[20];
int p =0;






/*
   Cabe�alho das fun��es auxiliares
 */
void inicializa_sinapses();
int gera_nr_aleatorios();
void mostrar_sinapses();
double f_sigmoid(double net);
void calcular_saidas(double entradas[ENTRADAS]);
void treinar_RNA();
double calcular_erro(double desejado, double saida);
void menu();
void calcular_delta_saida(double desejado);
void calcular_delta_oculta();
void calcular_gradiente_oculta();
void ajustar_pesos_sinapticos(double entradas[ENTRADAS]);
void gravar_pesos_sinapticos();
void restaurar_pesos_sinapticos();



/*
   Fun��o principal
 */
int main()
{
        srand(time(NULL));

        while (1) {

                menu();

        }

        return 0;
}

void inicializa_sinapses()
{
        int i, j;

        // Inicializa pesos sin�pticos da entrada para a camada oculta
        for (i = 0; i < ENTRADAS+1; i++) {
                for (j =0; j < NR_NEURON_O; j++) {
                        #if ZERO_TEST
                        w_e_o[i][j] = 0.0;
                        #else
                        w_e_o[i][j] = gera_nr_aleatorios();
                        #endif
                }
        }

        // Inicializa pesos sin�pticos da camada oculta para a sa�da
        for (i = 0; i < NR_NEURON_O+1; i++) {
                for (j =0; j < SAIDAS; j++) {
                        #if ZERO_TEST
                        w_o_s[i][j] = 0.0;
                        #else
                        w_o_s[i][j] = gera_nr_aleatorios();
                        #endif
                }
        }

}

int gera_nr_aleatorios()
{
        int numeros[2] = {-1, 1};
        // Retorna -1 ou 1

        return (numeros[rand() % 2]);
}

void mostrar_sinapses()
{
        int i, j;

        // Inicializa pesos sin�pticos da entrada para a camada oculta
        for (i = 0; i < ENTRADAS+1; i++) {
                for (j =0; j < NR_NEURON_O; j++) {
                        printf("w_e_o[%d][%d]: %f ", i, j, w_e_o[i][j]);
                }
                printf("\n");
        }

        // Inicializa pesos sin�pticos da camada oculta para a sa�da
        for (i = 0; i < NR_NEURON_O+1; i++) {
                for (j =0; j < SAIDAS; j++) {
                        printf("w_o_s[%d][%d]: %f ", i, j, w_o_s[i][j]);
                }
                printf("\n");
        }
}

double f_sigmoid(double net)
{
        return 1 / (1 + exp(-net));
}

void calcular_saidas(double entradas[ENTRADAS])
{
        int i, j;

        // Calcular os nets da entrada para a camada oculta
        for (i = 0; i < NR_NEURON_O; i++) {
                saida_o[i] = 0.0;
                saida_o[i] += w_e_o[0][i] * 1; // Calcula saida para bias

                for (j = 1; j < ENTRADAS+1; j++) {
                        saida_o[i] += w_e_o[j][i] * entradas[j-1];
                }

                // Calcular a saida de saida_o[i]
                saida_o[i] = f_sigmoid(saida_o[i]);
        }

        // Calcular os nets da camada oculta para a sa�da
        for (i = 0; i < SAIDAS; i++) {
                saida_s[i] = 0.0;
                saida_s[i] += w_o_s[0][i] * 1; // Calcula saida para bias

                for (j = 1; j < NR_NEURON_O+1; j++) {
                        saida_s[i] += w_o_s[j][i] * saida_o[j-1];
                }

                saida_s[i] = f_sigmoid(saida_s[i]);
        }

}

void treinar_RNA()
{
        int i, j;
        double entradas[ENTRADAS];

        for (i = 1; i <= EPOCAS; i++) {

                for (j = 0; j < NR_AMOSTRAS; j++) {
                        entradas[0] = cj_treinamento[j][0];
                        entradas[1] = cj_treinamento[j][1];
                        entradas[2] = cj_treinamento[j][2];
                        entradas[3] = cj_treinamento[j][3];
                        entradas[4] = cj_treinamento[j][4];

                    

                        // Feedforward
                        calcular_saidas(entradas);

                        // Backward (backpropagation)
                        calcular_delta_saida(cj_treinamento[j][5]/MAX_OUT);
                        calcular_gradiente_oculta();
                        calcular_delta_oculta();
                        ajustar_pesos_sinapticos(entradas);
                }

        }

}



double calcular_erro(double desejado, double saida)
{
        return desejado - saida;
}

void menu()
{

        int opcao,a;
        char letra[200] ={'0'};
        a=0;
        double entradas[ENTRADAS];
        
        printf("Rede Neural Perceptron de Multiplas Camadas\n");
        printf("Problema do OU EXCLUSIVO - XOR\n");
        printf("*******************************************\n\n");
        printf("1.Treinar a rede\n");
        printf("2.Usar a rede\n");
        printf("3.Ver pesos sinapticos\n");
        printf("4.Sair\n");
        printf("5. Mostrar frase\n");
        printf("Opcao? ");
        scanf("%d", &opcao);

        switch (opcao) {

        case 1: inicializa_sinapses();
                treinar_RNA();
                break;

        case 2:
        {
        
                printf("Entrada :");
                scanf(" %s", &letra);

            while (letra[a]!= '\0' ){
                char codigo[26]; //={'@','#','$','%','*','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k'};
                codigo[0] = '@'; //a
                codigo[1] = '#'; //b
                codigo[2] = '$'; //c
                codigo[3] = '%'; //d
                codigo[4] = '&'; //e
                codigo[5] = '*'; //f
                codigo[6] = '1'; //g
                codigo[7] = '2'; //  h
                codigo[8] = '3'; //i      
                codigo[9] = '4'; //j    
                codigo[10] = '5'; //k 
                codigo[11] = '6'; //l
                codigo[12] = '7'; //m
                codigo[13] = '8'; //n
                codigo[14] = '9'; //o
                codigo[15] = '0'; //p
                codigo[16] = 'a'; //q
                codigo[17] = 'b'; //r
                codigo[18] = 'c'; //s 
                codigo[19] = 'd'; //t
                codigo[20] = 'e'; //u
                codigo[21] = 'f'; //v
                codigo[22] = 'g'; //w
                codigo[23] = 'h'; //x
                codigo[24] = 'i'; //y
                codigo[25] = 'j'; //z */
                // printf("%d\n", letra);
                for(int i = 0; i < NR_AMOSTRAS; i++) {
                        if(codigo[i] == letra[a]) {
                                for(int j = 0; j < ENTRADAS; j++) {
                                        entradas[j] = cj_treinamento[i][j];
                                        //printf("%lf\n", entradas[j]);
                                }
                        }
                }

                calcular_saidas(entradas);
                //printf("\nResposta da RNA: %c \n", (int) round(saida_s[0] * MAX_OUT));
                frase[p]= (int) round(saida_s[0]*MAX_OUT);
                p++;

              a++;
                }
                break;

                

        }

        case 3:
                mostrar_sinapses();
                break;

        case 4: exit(0);

        
        case 5: 
            printf("Frase: %s\n", frase);
            break;
        }
    

}

void calcular_delta_saida(double desejado)
{

        int i;
        for (i = 0; i < SAIDAS; i++)
                delta_saida[i] = calcular_erro(desejado, saida_s[i]) * (1 - saida_s[i] * saida_s[i]);
}

void calcular_gradiente_oculta()
{
        int i, j;

        for (i = 0; i < SAIDAS; i++) {
                for (j = 1; j < NR_NEURON_O+1; j++) {
                        gradiente_oculta[j-1] = delta_saida[i] * w_o_s[j][i];
                }
        }
}

void calcular_delta_oculta()
{
        int i;
        for (i = 0; i < NR_NEURON_O; i++)
                delta_oculta[i] = gradiente_oculta[i] * saida_o[i] * (1 - saida_o[i]);
}

void ajustar_pesos_sinapticos(double entradas[ENTRADAS])
{
        int i, j;

        // Ajusta os pesos sin�pticos da camada intermedi�ria para a camada de sa�da
        for (i = 0; i < SAIDAS; i++) {
                w_o_s[0][i] = w_o_s[0][i] + TX_APRENDIZADO * delta_saida[i] * 1;

                for (j = 1; j < NR_NEURON_O+1; j++) {
                        w_o_s[j][i] = w_o_s[j][i] + TX_APRENDIZADO * delta_saida[i] * saida_o[j-1];
                }

        }

        // Ajusta os pesos sin�pticos da camada de saida para a camada intermedi�ria
        for (i = 0; i < NR_NEURON_O; i++) {
                w_e_o[0][i] = w_e_o[0][i] + TX_APRENDIZADO * delta_oculta[i] * 1;

                for (j = 1; j < ENTRADAS+1; j++) {
                        w_e_o[j][i] = w_e_o[j][i] + TX_APRENDIZADO * delta_oculta[i] * entradas[j-1];
                }
        }

}
