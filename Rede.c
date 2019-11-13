#include <stdio.h>
#include <stdlib.h>

#define DADOS 8
#define ENTRADAS 3
#define SAIDAS 2
#define N 0.7 //taxa de aprendizado
#define EPOCAS 1000

int dados[8][5] = 	{ 
	{0, 0, 0, 0, 0},
	{0, 0, 1, 0, 1},
	{0, 1, 0, 1, 1},
	{0, 1, 1, 1, 1},
	{1, 0, 0, 1, 0},
	{1, 0, 1, 1, 0},
	{1, 1, 0, 0, 0},
	{1, 1, 1, 0, 0},

};


float w1[ENTRADAS+1] = {0, 0, 0, 0}; // pesos sinapticos neuronio 1
float w2[ENTRADAS+1] = {0, 0, 0, 0}; // pesos sinapticos neuronio 2

int erro(int desejado, int saida);
int degrau(float x);
float soma(int entrada[3], int neu);
void print();
int saida(float soma);
void treinarRNA();
void pesos(int entrada[3], int erro, int neu);

int erro(int desejado, int saida){
	return desejado-saida;
}

int degrau(float x){
	if(x>=1) return 1;

	return 0;
}

float soma(int entrada[3], int neu){
	float soma=0;

	if(neu == 1){
		soma = w1[0] * 1; //bias
		for(int i=0;i<3;i++){
			soma+=w1[i+1]*entrada[i]; //Realizando a funçao soma para o neuronio 1
		}
	}else{
		soma= w2[0] * 1; //bias
		for(int i=0;i<3;i++){
			soma+=w1[i+1]*entrada[i]; //Realizando a funçao soma para o neuronio 2
		}

	}
	return soma;

}

void print(){
	for (int i = 0; i <= ENTRADAS; ++i){
		printf("Neuronio 1: w1[%d]: %.2f \n", i, w1[i] );
	}
	for (int i = 0; i <= ENTRADAS; ++i){
		printf("Neuronio 2: w2[%d]: %.2f \n", i, w2[i] );
	}
}

int saida(float soma){
	return degrau(soma);
}

void treinarRNA(){
	int i, j, entradas[3], saida1, saida2, erro1, erro2;
	for (int i = 0; i < EPOCAS; i++){
		for (int j = 0; j < DADOS; j++){
			entradas[0] = dados[j][0];
			entradas[1] = dados[j][1];
			entradas[2] = dados[j][2];

			saida1 = saida(soma(entradas, 1));
			saida2 = saida(soma(entradas, 0));
			erro1 = erro(dados[j][3], saida1);
			erro2 = erro(dados[j][4], saida2);
			if(erro1 != 0){
				pesos(entradas, erro1, 1);
			}
			if(erro2 != 0){
				pesos(entradas, erro2, 0);
			}
		}
	}
}


void pesos(int entrada[3], int erro, int neu){
	int i;
	if(neu == 1){
		w1[0] = w1[0] + N * erro * 1;
		for (int i = 1; i <= ENTRADAS; i++){
			w1[i] = w1[i] + N * erro * entrada[i-1];
		}

	}else{
		w2[0] = w2[0] + N * erro * 1;
		for (int i = 1; i <= ENTRADAS; i++){
			w2[i] = w2[i] + N * erro * entrada[i-1];
		}
	}
}

int main(){
	
	int entradas[3],resposta=1;



	treinarRNA();

	print();

	while(resposta==1){

		printf("Entrada 1: ");
		scanf("%d", &entradas[0]);
		printf("Entrada 2: ");
		scanf("%d", &entradas[1]);
		printf("Entrada 3: ");
		scanf("%d", &entradas[2]);

		printf("Saida 1 de RNA: %d\n", saida(soma(entradas, 1)));
		printf("Saida 2 de RNA: %d\n", saida(soma(entradas, 0)));

		printf("\nDeseja continuar? 1-Sim 2-Nao");
		scanf("%d", &resposta);
	} 


	return 0;
}