/*==================================================================
 * [P, it, Obj, Dif] = PFDR_graph_loss_d1_simplex_mex(Q, x, al, Eu, Ev, La_d1, rho, condMin, difRcd, difTol, itMax, verbose)
 * 
 * Q -> T al -> T Ru
 *  Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/PFDR_graph_loss_d1_simplex.hpp"
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <iostream> 
#include <string.h>
   
using namespace std; 


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int dims; 
    const mxArray *cell_Q, *cell_E, *cell_S, *cell_T, *cell_La, *cell_x, *cell_W;
    mxArray *cellArray_Q, *cellArray_E, *cellArray_S, *cellArray_T, *cellArray_La, *cellArray_x, *cellArray_W;
    mwIndex jcell;
    
    cell_Q = prhs[0];
    cell_x = prhs[1];
    cell_E = prhs[4];
    cell_S = prhs[4];
    cell_T = prhs[5];
    cell_La = prhs[6];
    cell_W = prhs[7];
    dims = mxGetNumberOfElements(prhs[0]); //Get number of channels
    // printf("Dims 0 = %d, 1 = %d",dims,dims);
    int nCh=dims;
    int K; //get number of classes
    int* V = (int*) malloc(K*sizeof(int)); //get cols in each array Q
    int* E = (int*) malloc(K*sizeof(int)); //get number of elements in Eu
    int** Eu = new int* [nCh];//get source of graph arrays
    int** Ev = new int* [nCh];//get target of graph arrays
    int V_full=0; //total number of points;


    for (jcell=0; jcell<nCh; jcell++) {
        cellArray_Q = mxGetCell(cell_Q,jcell);
        K=mxGetM(cellArray_Q);
        V[jcell]=round(mxGetN(cellArray_Q));
        V_full+=V[jcell];
        cellArray_E = mxGetCell(cell_E,jcell);//get jcell array
        E[jcell]=round(mxGetM(cellArray_E));
        cellArray_S = mxGetCell(cell_S,jcell);
        cellArray_T = mxGetCell(cell_T,jcell);
        mwSize s_size=mxGetNumberOfElements(cellArray_S);
        mwSize t_size=mxGetNumberOfElements(cellArray_T); 
        Eu[jcell]= new int [s_size];
        Ev[jcell]= new int [t_size];
        //[i][j] means imax =nCh, j max num elements
        int* S_arr = (int*) mxGetPr(cellArray_S);
        int* T_arr = (int*) mxGetPr(cellArray_T);
        int M=mxGetM(cellArray_S);
        int N=mxGetN(cellArray_S);
        for (int j=0;j<s_size;j++)
        {
            Eu[jcell][j]=S_arr[j];
            Ev[jcell][j]=T_arr[j];
            // cout<<jcell<<" "<<j<<" "<<Eu[jcell][j]<<" "<<Ev[jcell][j]<<endl;
        }
    }

    const int itMax = (int) mxGetScalar(prhs[12]);     //get max iterations
    const int verbose = (int) mxGetScalar(prhs[13]);    //get verbose value
    plhs[0] = mxCreateDoubleMatrix(K,V_full,mxREAL);

    //plhs[0] = mxDuplicateArray(prhs[0]);    //output Array initialized
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);   //1x1 size array
    int *it = (int*) mxGetData(plhs[1]);    

    if (mxIsDouble(prhs[0])){
        double **Q=new double*[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_Q = mxGetCell(cell_Q,jcell);
            mwSize Q_size=mxGetNumberOfElements(cellArray_Q);
            Q[jcell]= new double [Q_size];
            double* Q_arr = (double*) mxGetPr(cellArray_Q);
            int M=mxGetM(cellArray_Q);
            int N=mxGetN(cellArray_Q);
            for(int j=0; j<N; j++){
                for(int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<Q_arr[j*M+i]<<endl;
                    Q[jcell][j*M+i]=Q_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<Q[jcell][j*M+i]<<endl;
                }
            }
        }
        int offset=0;
        double *x=new double[V_full];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_x = mxGetCell(cell_x,jcell);
            mwSize x_size=mxGetNumberOfElements(cellArray_x);
            long long* x_arr = (long long*) mxGetPr(cellArray_x);
            int M=mxGetM(cellArray_x);
            int N=mxGetN(cellArray_x);

            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x_arr[j*M+i]<<endl;
                    x[j*M+i+offset]=x_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[(jcell*nCh+j)*M+i]<<endl;
                }
            }  
            offset+=V[jcell];
        }
        double **La_d1=new double*[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_La = mxGetCell(cell_La,jcell);
            mwSize La_size=mxGetNumberOfElements(cellArray_La);
            La_d1[jcell]= new double [La_size];
            double* La_arr = (double*) mxGetPr(cellArray_La);
            int M=mxGetM(cellArray_La);
            int N=mxGetN(cellArray_La);
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<La_arr[j*M+i]<<endl;
                    La_d1[jcell][j*M+i]=La_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[offset+j*M+i]<<endl;
                }
            }  
        } 
        
        const double unsymm_penalty = (double) mxGetScalar(prhs[2]);
        const double al = (double) mxGetScalar(prhs[3]);
        const double rho = (double) mxGetScalar(prhs[8]);
        const double condMin = (double) mxGetScalar(prhs[9]);
        const double difRcd = (double) mxGetScalar(prhs[10]);
        const double difTol = (double) mxGetScalar(prhs[11]);

        // plhs[0] = mxDuplicateArray(prhs[0]);
        offset=0;
        double *P=new double[V_full];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_Q = mxGetCell(cell_Q,jcell);
            mwSize Q_size=mxGetNumberOfElements(cellArray_Q);
            double* Q_arr = (double*) mxGetPr(cellArray_Q);
            int M=mxGetM(cellArray_Q);
            int N=mxGetN(cellArray_Q);
            //cout<<x_size<<" "<<M<<" "<<N<<endl;
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<j*M+i+offset<<" "<<Q_arr[j*M+i]<<endl;
                    P[j*M+i+offset]=Q_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[offset+j*M+i]<<endl;
                }
            }  
            offset+=V[jcell];
        }

        double *W=new double[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_W = mxGetCell(cell_W,jcell);
            double *W_arr = (double*)mxGetPr(cellArray_W);
            int M=mxGetM(cellArray_W);
            int N=mxGetN(cellArray_W);
            // cout<<" "<<M<<" "<<N<<endl;
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    W[jcell]=W_arr[j*M+i];
                }
            }  
        }

        // double *P = (double*) mxGetData(plhs[0]);
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
        // printf("Reached\n");
        PFDR_graph_loss_d1_simplex<double>(nCh, K, V, E, al, P, Q, x,W, Eu, Ev, unsymm_penalty, La_d1, \
                                           rho, condMin, difRcd, difTol, \
                                         itMax, it, Obj, Dif, verbose);
                                          // 18 arguments
    }else{
        float **Q=new float*[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_Q = mxGetCell(cell_Q,jcell);
            mwSize Q_size=mxGetNumberOfElements(cellArray_Q);
            Q[jcell]= new float [Q_size];
            double* Q_arr = (double*) mxGetPr(cellArray_Q);
            int M=mxGetM(cellArray_Q);
            int N=mxGetN(cellArray_Q);
            for(int j=0; j<N; j++){
                for(int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<Q_arr[j*M+i]<<endl;
                    Q[jcell][j*M+i]=Q_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<Q[jcell][j*M+i]<<endl;
                }
            }
        }
        
        int offset=0;
        float *x=new float[V_full];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_x = mxGetCell(cell_x,jcell);
            mwSize x_size=mxGetNumberOfElements(cellArray_x);
            long long* x_arr = (long long*) mxGetPr(cellArray_x);
            int M=mxGetM(cellArray_x);
            int N=mxGetN(cellArray_x);
            // cout<<x_size<<" "<<M<<" "<<N<<endl;
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x_arr[j*M+i]<<endl;
                    x[j*M+i+offset]=x_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[offset+j*M+i]<<endl;
                }
            }  
            offset+=V[jcell];
        }
        
        float **La_d1=new float*[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_La = mxGetCell(cell_La,jcell);
            mwSize La_size=mxGetNumberOfElements(cellArray_La);
            La_d1[jcell]= new float [La_size];
            float* La_arr = (float*) mxGetPr(cellArray_La);
            int M=mxGetM(cellArray_La);
            int N=mxGetN(cellArray_La);
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<La_arr[j*M+i]<<endl;
                    La_d1[jcell][j*M+i]=La_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[offset+j*M+i]<<endl;
                }
            }  
        }        
        const float unsymm_penalty = (float) mxGetScalar(prhs[2]);
        const float al = (float) mxGetScalar(prhs[3]);
        const float rho = (float) mxGetScalar(prhs[8]);
        const float condMin = (float) mxGetScalar(prhs[9]);
        const float difRcd = (float) mxGetScalar(prhs[10]);
        const float difTol = (float) mxGetScalar(prhs[11]);
        

        // plhs[0] = mxDuplicateArray(prhs[0]);
        offset=0;
        float *P=new float[V_full*K];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_Q = mxGetCell(cell_Q,jcell);
            mwSize Q_size=mxGetNumberOfElements(cellArray_Q);
            double* Q_arr = (double*) mxGetPr(cellArray_Q);
            int M=mxGetM(cellArray_Q);
            int N=mxGetN(cellArray_Q);
            // cout<<Q_size<<" "<<M<<" "<<N<<endl;
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<j*M+i+offset<<" "<<Q_arr[j*M+i]<<endl;
                    P[j*M+i+offset]=Q_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<x[offset+j*M+i]<<endl;
                }
            }  
            offset+=V[jcell];
        }

        float *W=new float[nCh];
        for (jcell=0; jcell<nCh; jcell++) {
            cellArray_W = mxGetCell(cell_W,jcell);
            double *W_arr = (double*)mxGetPr(cellArray_W);
            int M=mxGetM(cellArray_W);
            int N=mxGetN(cellArray_W);
            // cout<<" "<<M<<" "<<N<<endl;
            for (int j=0;j<N;j++)
            {
                for (int i=0; i<M; i++)
                {
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<(jcell*nCh+j)*M+i<<endl;
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<j*M+i+offset<<" "<<Q_arr[j*M+i]<<endl;
                    W[jcell]=W_arr[j*M+i];
                    // cout<<jcell<<" "<<j<<" "<<i<<" "<<W_arr[j*M+i]<<endl;
                }
            }  
            // cout<<jcell<<" "<<W[jcell]<<endl;
        }

        // offset=0;
        // for (jcell=0; jcell<nCh; jcell++) {
        //     int V_val=V[jcell];
        //     cellArray_Q = mxGetCell(cell_Q,jcell);
        //     memcpy(mxGetPr(plhs[0])+offset,cellArray_Q,V_val*sizeof(float));  
        //     offset+=V_val;
        // }

        // float *P = (float*) mxGetData(plhs[0]);
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

        // printf("Reached\n");
        PFDR_graph_loss_d1_simplex<float>(nCh, K, V, E, al, P, Q, x,W, Eu, Ev, unsymm_penalty, La_d1, \
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
