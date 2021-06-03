#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

//Reserva memoria para una matriz de m x n
double **createMatrix(int m, int n) {
	double **A = (double **)malloc(m * sizeof(double *));
	A[0] = (double *)calloc(m * n, sizeof(double));
	for (int i = 1; i < m; i++)
		A[i] = A[i - 1] + n;
	return A;
}

//Libera la memoria de una matriz
void freeMatrix(double **A) {
	free(A[0]);
	free(A);
}

//Reserva memoria para una matriz 3d de l x m x n
double ***create3dMatrix(int l, int m, int n) {
	double ***A = (double ***)malloc(l * sizeof(double **));
	A[0] = createMatrix(l * m, n);
	for (int i = 1; i < l; i++)
		A[i] = A[i - 1] + m;
	return A;
}

void free3dMatrix(double ***A) {
	freeMatrix(A[0]);
	free(A);
}

//Transfiere los datos de una imagen a una matriz 3d de doubles
void imgTo3dMatrix(int ndatos, unsigned char *img, double ***A) {
	for (int i = 0; i < ndatos; i++)
		A[0][0][i] = img[i];
}

//Transfiere los datos de una matriz 3d de doubles a una imagen
void imgFrom3dMatrix(int ndatos, unsigned char *img, double ***A) {
	for (int i = 0; i < ndatos; i++)
		img[i] = A[0][0][i];
}

/* Difusion Anisotropica para una imagen en escala de grises
   Argumentos:
     height - altura de la imagen
     width  - ancho de la imagen
     comp   - numero de componentes de la imagen (1 para escala de grises y 3 para RGB)
     img    - datos de la imagen
     niter  - numero de iteraciones
     option - coeficientes de difusion:
  	 	 0: 1/(1 + (|grad I|/K)^2)    1: e^(-(|grad I|/K)^2)
     K      - constante utilizada para el coeficiente de difusion
     dx     - distancia horizontal entre pixeles
     dy     - distancia vertical entre pixeles
     dt     - distancia entre una iteracion y otra
   La funcion modifica los valores de la matriz.
*/
void AnisotropicDiffusion(int height, int width, int comp, double ***img, int niter, bool option, double K, double dx, double dy, double dt) {
	//Matrices auxiliares
	double **dIdx = createMatrix(height, width);
	double **dIdy = createMatrix(height, width);
	double **divX = createMatrix(height, width);
	double **divY = createMatrix(height, width);

	while (niter--) {
		for (int k = 0; k < comp; k++) {
			//Calcula el gradiente de la imagen
			for (int i = 0; i < height; i++)
				for (int j = 1; j < width - 1; j++)
					dIdx[i][j] = (img[i][j + 1][k] - img[i][j - 1][k]) / (2*dx);
			for (int i = 1; i < height - 1; i++)
				for (int j = 0; j < width; j++)
					dIdy[i][j] = (img[i + 1][j][k] - img[i - 1][j][k]) / (2*dy);

			//Calcula c * grad(I)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++) {
					double c = option ? exp(-(dIdx[i][j]*dIdx[i][j] + dIdy[i][j]*dIdy[i][j])/(K*K))
							  : 1.0/(1 + (dIdx[i][j]*dIdx[i][j] + dIdy[i][j]*dIdy[i][j])/(K*K));
					dIdx[i][j] *= c;
					dIdy[i][j] *= c;
				}

			//Calcula la divergencia de (c * grad(I))
			for (int i = 0; i < height; i++)
				for (int j = 1; j < width - 1; j++)
					divX[i][j] = (dIdx[i][j + 1] - dIdx[i][j - 1]) / (2*dx);
			for (int i = 1; i < height - 1; i++)
				for (int j = 0; j < width; j++)
					divY[i][j] = (dIdy[i + 1][j] - dIdy[i - 1][j]) / (2*dy);

			//Actualiza la imagen
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					img[i][j][k] += dt * (divX[i][j] + divY[i][j]);
		}
	}

	freeMatrix(dIdx);
	freeMatrix(dIdy);
	freeMatrix(divX);
	freeMatrix(divY);	
}


int main(int narg, char *argv[]) {
	//Lee la imagen
	int height, width, comp;
	unsigned char *img = stbi_load(argv[1], &height, &width, &comp, 0);

	//Transfiere los datos de la imagen a una matriz
	double ***imgDouble = create3dMatrix(height, width, comp);
	imgTo3dMatrix(height*width*comp, img, imgDouble);

	//Difusion Anisotropica
	int niter = atoi(argv[3]); 
	bool option = atoi(argv[4]);
	double K  = atof(argv[5]); 
	double dx = atof(argv[6]);
	double dy = atof(argv[7]); 
	double dt = atof(argv[8]); 
	AnisotropicDiffusion(height, width, comp, imgDouble, niter, option, K, dx, dy, dt);

	//Transfiere los datos de la matriz a la imagen.
	imgFrom3dMatrix(height*width*comp, img, imgDouble);
	stbi_write_jpg(argv[2], height, width, comp, img, 100);	

	//Libera memoria
	stbi_image_free(img);
	free3dMatrix(imgDouble);
	return 0;
}