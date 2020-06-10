/*==================================================================
 * [P, it, Obj, Dif] = PFDR_graph_loss_d1_simplex_mex(Q, x, al, Eu, Ev, La_d1, rho, condMin, difRcd, difTol, itMax, verbose)
 * 
 * Q -> T al -> T Ru
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/PFDR_graph_loss_d1_simplex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     
    const int K = mxGetM(prhs[0]); //get rows in array Q
    const int V = mxGetN(prhs[0]);  //get cols in array Q
    const int E = mxGetNumberOfElements(prhs[4]);   //get number of elements in Eu
    const int *Eu = (int*) mxGetData(prhs[4]);      //get source of graph array
    const int *Ev = (int*) mxGetData(prhs[5]);      //get target of the graph array
    const int itMax = (int) mxGetScalar(prhs[11]);     //get max iterations
    const int verbose = (int) mxGetScalar(prhs[12]);    //get verbose value
    plhs[0] = mxDuplicateArray(prhs[0]);    //output Array initialized
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);   //1x1 size array
    int *it = (int*) mxGetData(plhs[1]);    

    
    if (mxIsDouble(prhs[0])){
        const double *Q = (double*) mxGetData(prhs[0]);
        const double *x = (double*) mxGetData(prhs[1]);
        const double unsymm_penalty = (double) mxGetScalar(prhs[2]);
        const double al = (double) mxGetScalar(prhs[3]);
        const double *La_d1 = (double*) mxGetData(prhs[6]);
        const double rho = (double) mxGetScalar(prhs[7]);
        const double condMin = (double) mxGetScalar(prhs[8]);
        const double difRcd = (double) mxGetScalar(prhs[9]);
        const double difTol = (double) mxGetScalar(prhs[10]);

        plhs[0] = mxDuplicateArray(prhs[0]);
        double *P = (double*) mxGetData(plhs[0]);
        double *Obj = NULL;
        if (nlhs > 2){
            plhs[2] = mxCreateNumericMatrix(1, itMax+1, mxDOUBLE_CLASS, mxREAL);
            Obj = (double*) mxGetData(plhs[2]);
        }
        double *Dif = NULL;
        if (nlhs > 3){
            plhs[3] = mxCreateNumericMatrix(1, itMax, mxDOUBLE_CLASS, mxREAL);
            Dif = (double*) mxGetData(plhs[3]);
        }

        PFDR_graph_loss_d1_simplex<double>(K, V, E, al, P, Q, x, Eu, Ev, unsymm_penalty, La_d1, \
                                           rho, condMin, difRcd, difTol, \
                                         itMax, it, Obj, Dif, verbose);
                                          // 18 arguments
    }else{
        const float *Q = (float*) mxGetData(prhs[0]);
        const float *x = (float*) mxGetData(prhs[1]);
        const float unsymm_penalty = (float) mxGetScalar(prhs[2]);
        const float al = (float) mxGetScalar(prhs[3]);
        const float *La_d1 = (float*) mxGetData(prhs[6]);
        const float rho = (float) mxGetScalar(prhs[7]);
        const float condMin = (float) mxGetScalar(prhs[8]);
        const float difRcd = (float) mxGetScalar(prhs[9]);
        const float difTol = (float) mxGetScalar(prhs[10]);
        
        plhs[0] = mxDuplicateArray(prhs[0]);
        float *P = (float*) mxGetData(plhs[0]);
        float *Obj = NULL;
        if (nlhs > 2){
            plhs[2] = mxCreateNumericMatrix(1, itMax+1, mxSINGLE_CLASS, mxREAL);
            Obj = (float*) mxGetData(plhs[2]);
        }
        float *Dif = NULL;
        if (nlhs > 3){
            plhs[3] = mxCreateNumericMatrix(1, itMax, mxSINGLE_CLASS, mxREAL);
            Dif = (float*) mxGetData(plhs[3]);
        }

        PFDR_graph_loss_d1_simplex<float>(K, V, E, al, P, Q, x, Eu, Ev, unsymm_penalty, La_d1, \
                                        rho, condMin, difRcd, difTol, \
                                          itMax, it, Obj, Dif, verbose);
                                       //    18 arguments 
    }
    // check inputs
   /* mexPrintf("K = %d, V = %d, E = %d, al = %g, P[0] = %f, Q[0] = %f\n \
    Eu[0] = %d, Ev[0] = %d, La_d1[0] = %g, rho = %g, condMin = %g\n \
    difRcd = %g, difTol = %g, itMax = %d, *it = %d\n \
    objRec = %d, difRec = %d, verbose = %d\n", \
    K, V, E, al, P[0], Q[0], Eu[0], Ev[0], La_d1[0], rho, condMin, \
    difRcd, difTol, itMax, *it, Obj != NULL, Dif != NULL, verbose);*/
    //mexEvalString("pause");
    
}
