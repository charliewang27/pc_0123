#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */

/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/

/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */

void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}

/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}

/************************************************************/

int main(int argc, char *argv[])
{

 int i, commsize, myrank;
 int nit = 0; /* number of iterations */
 int converged;

 if( argc != 2)
 {
   printf("Usage: gsref filename\n");
   exit(1);
 }
  
 /* Read the input file and fill the global data structure above */ 
 get_input(argv[1]);
 
 /* Check for convergence condition */
 /* This function will exit the program if the coffeicient will never converge to 
  * the needed absolute error. 
  * This is not expected to happen for this programming assignment.
  */
 check_matrix();

 MPI_Init(&argc, &argv);
 MPI_Comm_Size(MPI_COMM_WORLD, &commsize);
 MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

 int n_lim = num/commsize;
 
 int countarr[commsize], displsarr[commsize], int receivarr[commsize];
 int disp = 0;


 for (int i = 0; i < commsize; i++){
  countarr[i] = n_lim;
  receivarr[i] = countarr[i];
  displsarr[i] = disp;
  disp = disp + countarr[i];
 }

 //Send count for MPI Scatter
 int send_count = n_lim * num;

 float *localX = (float * ) malloc (n_lim *sizeof(float));
 float *localA = (float *) malloc(send_count * sizeof(float));
 float *localB = (float *) malloc(n_lim * sizeof(float));
 float *curr = (float *) malloc(num * sizeof(float));
 float *localD = (float *) malloc(n_lim * sizeof(float));

 MPI_Scatter(a, send_count, MPI_FLOAT, localA, send_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatter(b, localNum, MPI_FLOAT, localB, localNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatter(diag, localNum, MPI_FLOAT, localD, localNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatterv(x, countarr, displsarr, MPI_FLOAT, localX, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

 for(i = 0; i < num; i++){
  curr[i] = x[i];
 }

 do{
  nit++;
  for(i = 0; i < num; i++){
    x[i] = curr[i];
  }

  for(i = 0; i < countarr[my_rank]; i++){
    
    int global_i = i;
    int k;
    for(k = 0; k < my_rank; k++){
      global_i += countarr[k];
    }   
    localX[i] = localB[i];
    int j;
    for(j = 0; j < global_i; j++){
      localX[i] = localX[i] - localA[i * num + j] * x[j];
    }
    for(j = global_i + 1; j < num; j++){
      localX[i] = localX[i] - localA[i * num + j] * x[j];
    }
    localX[i] = localX[i]/localD[i];
  }
  MPI_Allgatherv(localX, countarr[my_rank], MPI_FLOAT, curr, recv, displsarr, MPI_FLOAT, MPI_COMM_WORLD);
 }while(checkErr(curr, num));

 if( my_rank == 0){
/* Writing to the stdout */
/* Keep that same format */
  for(i = 0; i < num; i++){
    printf("%f\n", x[i]);
  }
  printf("total number of iterations: %d\n", nit);

  free(x);
  free(a);
  free(b);
  free(diag);
 } 

 free(localX);
 free(localA);
 free(localB);
 free(curr);
 free(localD);

 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Finalize();

 /* Writing to the stdout */
 /* Keep that same format */
 for( i = 0; i < num; i++)
   printf("%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 
 exit(0);

}
