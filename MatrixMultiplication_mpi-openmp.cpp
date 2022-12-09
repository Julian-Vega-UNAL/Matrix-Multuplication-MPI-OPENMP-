#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

#include "omp.h"
#include "mpi.h"

using namespace std;
void print_matrix(float *matrix, int size, char name){
    cout << name << ":" << endl;
    for (int i = 0; i < size; i++){
        cout << "[ ";
        for (int j = 0; j < size; j++){
            cout << matrix[i*size + j] << ", ";
        }
        cout << "]," << endl;
    }
    cout << endl;
}
int main (int argc, char *argv[])
{
    int n = stoi(argv[1]);
    int THREADS = stoi(argv[2]);
    int sizeA = n*n*sizeof(float);
    int size, rank, rows, init;
    float result;

    srand(time(0));

    float *A = (float *) malloc(sizeA);
    float *B = (float *) malloc(sizeA);
    float *C = (float *) malloc(sizeA);
    float *cpartial = (float *) malloc(sizeA);
    MPI_Status status;
    init = MPI_Init(&argc, &argv);
    if (init != MPI_SUCCESS)
        {
                printf ("Hubo un error al inociar el proceso. \n");
                MPI_Abort (MPI_COMM_WORLD, init);
        }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                A[i*n + j] = (float) rand()/RAND_MAX;
                B[i*n + j] = (float) rand()/RAND_MAX;
                C[i*n + j] = 0;
                cpartial[i*n + j] = 0;
            }
        }
        print_matrix(A, n, 'A');
        print_matrix(B, n, 'B');
        rows = n / size;
    }

    MPI_Bcast (&n, 1 , MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast (&rows, 1 , MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast (B, n * n , MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast (cpartial, n * n , MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows*n, MPI_FLOAT, A, rows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);


    omp_set_num_threads(THREADS);
    int i, j, k;
    #pragma omp parallel private(i,j,k)
    {
        #pragma omp for schedule(dynamic,2)
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                for (int k = 0; k < n; k++){
                    cpartial[i*n + j] += A[i*n + k] * B[k*n + j];
                }
            }
        }
    }

    MPI_Gather(cpartial, rows*n, MPI_FLOAT, C, rows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        //print_matrix(C, n, 'C');
    }
    free(A);
    free(B);
    free(C);
    free(cpartial);
    MPI_Finalize();
}