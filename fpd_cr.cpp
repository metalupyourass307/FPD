// fpd_cr_variable_ph.cpp
// MODIFIED to correctly handle both weak acid and weak base colloids
// by accepting pKa and pKb as separate command-line arguments.

//Copyright (c) 2022 Kyohei Takae
//All rights reserved.
//You can edit the code for your works collaborating with K.T.
//use and/or copy a part of the code for other purpose, and distributing with any other people, are not permitted.
//type 1 colloid has minus charge and type 2 colloid has positive charge
/***************include***************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <float.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>
#include <new>
#include <fftw3.h>
#include <iostream>
#include <fstream>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>

/*********** define ***************/
#define NUMBER_THREADS 1
#define M    128 //mesh size 
#define Nx   M
#define Ny   M
#define Nz   M
#define NX   (Nx+2) //PBC 
#define NY   (Ny+2) //PBC 
#define NZ   (Nz+2) //PBC 
#define Ncp    1 //number of negative colloids type 1
#define Ncp2    1 //number of positive colloids type 2
#define N    (Ncp+Ncp2) //number of colloids
#define adj  120 //number of contact colloids within cutoff
#define RADI 3.2 //colloid radius
#define XI   1 //finite interface in FPD
#define RD   ((int)(RADI+0.99)+2*XI+2) //compact size
#define RD2  (2*RD+1)
#define RR   (N+1)
#define DNUM 1
#define BUF_SIZE 256
#define LOAD  0
#define LOADL 0
#define noise 0 //0: noise off; 1: noise on
FILE *fr;
FILE *fw;
FILE *fp;
FILE *frho;
FILE *fvel;
const char *filename;
//char bif[]="R";//F: FCC, H: HCP, B: BCC, R: random
//int branch=-1;//-10:BD, -1:init, -2:initD(N<=2), -3:initv(N<=2), -4:Eq(N>10), 0:execute, 1:load and continue 10:Stokes approximation
//double ran_st=0.0;
double l0=10.0; //grid size in nm 
double scale=0.3/l0;//v_0^{1/3} / \ell_0, v_0^{1/3}=0.3 nm
double Bje=0.7/l0;//Bjerrum length of the solvent and the colloids: water = 0.7nm
double beta=1.0;
double inv_beta=1.0/beta;
//double Nd1=2.0,Nd2=Nd1;//density of surface charge
//double Nd1=5492.794403/(4.0*M_PI*RADI*RADI),Nd2=-Nd1;//density of surface charge
//double con=0.5;//concentration
double n0=0.0;//monovalent salt concentration
//double Dc=0.24*l0;//ion diffusivity 10^-5cm^2/s is assumed. dimensionless D=D/V0/l0
//double Dc2=Dc;
double exE=0.2,exE0=0.0;//external electric field and electric potential
double fc=0.0;//external mechanical force
int t_load=0;
//char dirP3[BUF_SIZE]="171018";

//double dsep2=50.0;//initial separation of two colloids


const double xi=(double)XI;
const double invxi=1.0/xi;
//const double Hxi=0.5*xi;
double scale3=scale*scale*scale,invscale3=1.0/scale3,lnscale=3.0*log(scale);
double Re=4.14195e-04;//Reynolds number
double gc=15.0;//selective solvation (gc*cc*phi)
//fixed parameters and derivative parameters
//double D1=Dc,D2=Dc;//ion diffusivity
double fc0=0.0;//for load

const double radi=(double)RADI;
double radi2=(radi+2.0*xi+2.1)*(radi+2.0*xi+2.1);
double radi2m=(radi-2.0*xi-2.1)*(radi-2.0*xi-2.1);
//double Nc1=Nd1*4.0*M_PI*(double)(RADI*RADI),Nc2=Nd2*4.0*M_PI*(double)(RADI*RADI);//number of ionizable monomer in a colloid
//double rx0int=0.0,rx0int2=2.0*radi+5.0*xi;


//reference Debye length
const double PI=M_PI,PIo2=0.5*PI,PI2=2.0*M_PI,PI4=2.0*PI2;
double Volume=(double)(Nx*Ny*Nz),invVol=1.0/Volume;
//double nc0=0.2*invVol*Nc1*(double)N + 2.0*n0;//20% ionization is assumed
//double kDH0=sqrt(PI4*Bje*nc0);
//double ka0=kDH0*radi;

//interparticle force
//double Hama=100.0;//Hamaker constant
//double sc=0.0035*Hama;//strength of soft core repulsion
double scr=1.0;//potential
double eM=5.0,wM=33.1;//Morse potential
const double sig=2.0*radi,sig2=sig*sig;
const double r0=1.12246204831*sig,r02=r0*r0;//LJ cutoff 1.12246204831
//double Hama2=Hama/sig2/6.0,sc2=4.0*sc/sig2;
double Fene=0.0;


/*********** CR Model Parameters ************/
double K_acid;       // ADD: Helper constant for acid dissociation
double K_base;       // ADD: Helper constant for base dissociation

/*********** global variable ************/
double t=0,quit_time=0;
double invN=1.0/(double)N;
double max,min,ave,vari;
int step=0;
double DT=2.5e-3;
int interval_pos=int(1/DT);
int interval_field=int(1000/DT);
double Nc[N];           // REPURPOSE: Now stores the CURRENT regulated charge of each colloid
double N_sites[N];      // ADD: Stores the MAXIMUM number of ionizable sites (read from file)
//double HDT=0.5*DT,DX=1.0,ph=0.5*DT/(DX*DX);
char save_dir[BUF_SIZE];
double Lx=(double)Nx,Ly=(double)Ny,Lz=(double)Nz;
double HLx=0.5*Lx,HLy=0.5*Ly,HLz=0.5*Lz;
int numid=1;
double tanh18=0.5-0.5*tanh(1.8),dtanh18=-0.5*invxi*(1.0-tanh(1.8)*tanh(1.8));
double tanc1=tanh18+0.1*dtanh18;
double tanc2=1.0/(1.0-2.0*tanc1);
double dtan1=-2.5*dtanh18*tanc2;
double invep=PI4*Bje;
double etal=1.0,etas=50.0*etal;//viscosity
//VSLStreamStatePtr stream;//intel mkl random number
/*----------colloids----------*/
double rx[N],ry[N],rz[N];//particle position and velocity
double rx_old[N],ry_old[N],rz_old[N];
double vdx[N],vdy[N],vdz[N];
double Fdx[N],Fdy[N],Fdz[N];
double invpvc[N],invpvx[N],invpvy[N],invpvz[N],invpbvc[N],invpbvx[N],invpbvy[N],invpbvz[N];
double phic[N][RD2][RD2][RD2],phix[N][RD2][RD2][RD2],phiy[N][RD2][RD2][RD2],phiz[N][RD2][RD2][RD2];
double phixy[N][RD2][RD2][RD2],phiyz[N][RD2][RD2][RD2],phizx[N][RD2][RD2][RD2];
double dphic[N][RD2][RD2][RD2],dphix[N][RD2][RD2][RD2],dphiy[N][RD2][RD2][RD2],dphiz[N][RD2][RD2][RD2];
double *Fdxid,*Fdyid,*Fdzid;
int plabel[N];
/*----------ions and electrostatics----------*/
double cp[NX][NY][NZ],cc[NX][NY][NZ],c1[NX][NY][NZ],c2[NX][NY][NZ];
double dcc[NX][NY][NZ],dc1[NX][NY][NZ],dc2[NX][NY][NZ];
double muc[NX][NY][NZ],mu1[NX][NY][NZ],mu2[NX][NY][NZ];
//double Lc[NX][NY][NZ],L1[NX][NY][NZ],L2[NX][NY][NZ];
double rho[NX][NY][NZ],rhot[NX][NY][NZ],rhop[NX][NY][NZ];
static double phit2[NX][NY],vx2[NX][NY],vy2[NX][NY],cpd[NX][NY],ccd[NX][NY],c1d[NX][NY],c2d[NX][NY];
double Z[N];
//binary mixture
//int N2=(int)(con*(double)N);
//int N1=N-N2;
//double invN1=1.0/(double)N1;
double cp2[NX][NY][NZ],cc2[NX][NY][NZ],dcc2[NX][NY][NZ],muc2[NX][NY][NZ];
// ADD: Arrays to store the maximum possible fixed charge density distribution
double cp_max[NX][NY][NZ], cp2_max[NX][NY][NZ];
//for salt
double csp[NX][NY][NZ],csm[NX][NY][NZ];
static double cp2d[NX][NY],cc2d[NX][NY];
/*----------fluid particle----------*/
double vx[NX][NY][NZ],vy[NX][NY][NZ],vz[NX][NY][NZ];//,vx_x[NX][NY],vx_y[NX][NY],vy_x[NX][NY],vy_y[NX][NY];
double phit[NX][NY][NZ],phixyt[NX][NY][NZ],phiyzt[NX][NY][NZ],phizxt[NX][NY][NZ];//shape function
double Fx[NX][NY][NZ],Fy[NX][NY][NZ],Fz[NX][NY][NZ];//force at points
double sigxx[NX][NY][NZ],sigyy[NX][NY][NZ],sigzz[NX][NY][NZ];
double sigxy[NX][NY][NZ],sigyz[NX][NY][NZ],sigzx[NX][NY][NZ];
//velocity update by MAC method
double P[NX][NY][NZ];
double vxd[NX][NY][NZ],vyd[NX][NY][NZ],vzd[NX][NY][NZ];
/*----------electrostatic poisson equation----------*/
double U[NX][NY][NZ];
/*----------FFT----------*/
double errorP[Nx][Ny][Nz],dP[Nx][Ny][Nz];
double errorU[Nx][Ny][Nz],dU[Nx][Ny][Nz];
//fftw
const int HNz=Nz/2;
const int Nz2=HNz+1;
double *fftw_in;
fftw_complex *fftw_out;
int fftw_init_threads(void);
/*----------region partitioning method----------*/
const double r1=r0+0.5*sig,r1_2=r1*r1;
//int Ncx=(int)(Lx/r1-0.001);
//int Ncy=(int)(Ly/r1-0.001);
//int Ncz=(int)(Lz/r1-0.001);
//int Nc3=Ncx*Ncy*Ncz;
int connect[N][adj];//connect[i][*]=the label of particle connected to i-th particle
/*************function & class prototype*************/
extern inline double shape_tanh(double r);
extern inline double shape_SP(double r);
extern inline double Laplacian1(int i,int j,int k,double A[NX][NY][NZ]);
extern inline double Laplacian2(int i,int j,int k,double A[NX][NY][NZ]);
extern inline double Divgrad1(int i,int j,int k,double A[NX][NY][NZ],double B[NX][NY][NZ]);
extern inline double Divgrad2(int i,int j,int k,double A[NX][NY][NZ],double B[NX][NY][NZ]);
extern inline double advection(int i,int j,int k,double A[NX][NY][NZ]);
extern void poisson(double A[Nx][Ny][Nz],double B[Nx][Ny][Nz]);
void initial(int);
void Cellupdate();
void calc_force();
void charge();
void write_data();
void write_potential();
void write_rho();
void update_electric_equilibrium();
void update_electric_field();
void update_shape_function();
void update_max_charge_density();
extern void update_velocity_particle();
extern void update_velocity_MAC();
void periodic_boundary(double A[NX][NY][NZ]);
void maxmin(double *A),maxmin2(double A[NX][NY][NZ]);
/*--------------------------------------------------------------
----------------------------------------------------------------
----------------------------------------------------------------
------------------------- main routine -------------------------
----------------------------------------------------------------
----------------------------------------------------------------
--------------------------------------------------------------*/































/*--------------------------------------------------------------
----------------------------------------------------------------
----------------------------------------------------------------
------------------------- sub routine -------------------------
----------------------------------------------------------------
----------------------------------------------------------------
--------------------------------------------------------------*/
/***************max, min, average and variance***************/
void maxmin(double *A){
  register int i;
  min=100;max=-100;ave=0;vari=0;
#pragma omp parallel for reduction(+:ave,vari) reduction(max:max) reduction(min:min)
  for(i=0;i<N;i++){
    double Ai=A[i];
    if(Ai>max){max=Ai;}
    if(Ai<min){min=Ai;}
    ave+=Ai;
    vari+=Ai*Ai;
  }
  ave=ave*invN;
  vari=vari*invN-ave*ave;
}
void maxmin2(double A[NX][NY][NZ]){
  register int i;
  min=100;max=-100;ave=0;vari=0;
#pragma omp parallel for reduction(+:ave,vari) reduction(max:max) reduction(min:min)
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double Aijk=A[i][j][k];
        if(Aijk>max){max=Aijk;}
        if(Aijk<min){min=Aijk;}
        ave+=Aijk;
        vari+=Aijk*Aijk;
      }
    }
  }
  ave=ave*invVol;
  vari=vari*invVol-ave*ave;
}
/***************advection term***************/
//this is central differences, may need to refine (e.g., WENO5)
inline double advection(int i,int j,int k,double A[NX][NY][NZ]){
  double Aijk=A[i][j][k];
  double adA=-0.5*(vx[i][j][k]*(A[i+1][j][k]-Aijk)+vx[i-1][j][k]*(Aijk-A[i-1][j][k])
                  +vy[i][j][k]*(A[i][j+1][k]-Aijk)+vy[i][j-1][k]*(Aijk-A[i][j-1][k])
                  +vz[i][j][k]*(A[i][j][k+1]-Aijk)+vz[i][j][k-1]*(Aijk-A[i][j][k-1]));
  return adA;
}
/***************solve electrostatic Poisson equation***************/
void update_electric_field(){
  register int i;
  //FFT
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        dU[i-1][j-1][k-1]=-invep*rho[i][j][k];
      }
    }
  }
  poisson(errorU,dU);
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int k=0;k<Nz;k++){
        U[i+1][j+1][k+1]=errorU[i][j][k];
      }
    }
  }
  maxmin2(U);
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        U[i][j][k]-=ave;
      }
    }
  }
  periodic_boundary(U);
  
}
/***************shape function***************/
inline double phi_tanh(double r){
  //r=distance-radi
  double rr=invxi*r;
  if(rr<-2.0){return 1.0;}
  else if(rr>2.0){return 0.0;}
  else if(rr<-1.8){return 1.0-dtan1*(rr+2.0)*(rr+2.0);}
  else if(rr>1.8){return dtan1*(rr-2.0)*(rr-2.0);;}
  else{return tanc2*(0.5*(1.0-tanh(rr)) - tanc1);}
}
//used in Takae sensei's soft matter paper eq(A1)
inline double dphi_tanh(double r){
  //r=distance-radi
  double rr=invxi*r;
  if(rr<-2.0){return 0.0;}
  else if(rr>2.0){return 0.0;}
  else if(rr<-1.8){return -2.0*dtan1*(rr+2.0);}
  else if(rr>1.8){return 2.0*dtan1*(rr-2.0);}
  else{
    double th=tanh(rr);
    return -tanc2*0.5*invxi*(1.0-th*th);
  }
}
//charge distribued in colloid center
/*inline double dphi_tanh(double r){
  //r=distance-radi
  double smaller_charge_radi=1.0;
  double rr=invxi*(r+radi-smaller_charge_radi);
  double th=0.5*(tanh(-rr)+1);
  return th;
}*/

void update_shape_function(){

    register int n,i;
  

  #pragma omp parallel for
    for(i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          phit[i][j][k]=0.0;
          phixyt[i][j][k]=0.0;
          phiyzt[i][j][k]=0.0;
          phizxt[i][j][k]=0.0;
        }
      }
    }

    for(n=0;n<N;n++){

  #pragma omp parallel for
      for(i=0;i<RD2;i++){
        for(int j=0;j<RD2;j++){
          for(int k=0;k<RD2;k++){
            phic[n][i][j][k]=0.0;
            phix[n][i][j][k]=0.0;
            phiy[n][i][j][k]=0.0;
            phiz[n][i][j][k]=0.0;
            phixy[n][i][j][k]=0.0;
            phiyz[n][i][j][k]=0.0;
            phizx[n][i][j][k]=0.0;
            dphic[n][i][j][k]=0.0;
            dphix[n][i][j][k]=0.0;
            dphiy[n][i][j][k]=0.0;
            dphiz[n][i][j][k]=0.0;
          }
        }
      }
      
      double pvolc=0.0,pvolx=0.0,pvoly=0.0,pvolz=0.0,pvolxy=0.0,pvolyz=0.0,pvolzx=0.0;
      double dpvolc=0.0,dpvolx=0.0,dpvoly=0.0,dpvolz=0.0;
      double rxn=rx[n];
      double ryn=ry[n];
      double rzn=rz[n];
      int RX=(int)rxn;
      int RY=(int)ryn;
      int RZ=(int)rzn;
  
  #pragma omp parallel for reduction(+:pvolc,pvolx,pvoly,pvolz,pvolxy,pvolyz,pvolzx,dpvolc,dpvolx,dpvoly,dpvolz)
      for(i=0;i<RD2;i++){
        int ii=RX+i-RD;
        double drx1=(double)ii-rxn;
        double drx2=drx1+0.5;
        double drx1_2=drx1*drx1;
        double drx2_2=drx2*drx2;
        for(int j=0;j<RD2;j++){
          int jj=RY+j-RD;
          double dry1=(double)jj-ryn;
          double dry2=dry1+0.5;
          double dry1_2=dry1*dry1;
          double dry2_2=dry2*dry2;
          for(int k=0;k<RD2;k++){
            int kk=RZ+k-RD;
            double drz1=(double)kk-rzn;
            double drz2=drz1+0.5;
            double drz1_2=drz1*drz1;
            double drz2_2=drz2*drz2;
            double dist2;
  
            dist2=drx1_2+dry1_2+drz1_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phic[n][i][j][k]=phi_tanh(dist);
              dphic[n][i][j][k]=dphi_tanh(dist);
              pvolc+=phic[n][i][j][k];
              dpvolc+=fabs(dphic[n][i][j][k]);
            }
            dist2=drx2_2+dry1_2+drz1_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phix[n][i][j][k]=phi_tanh(dist);
              dphix[n][i][j][k]=dphi_tanh(dist);
              pvolx+=phix[n][i][j][k];
              dpvolx+=fabs(dphix[n][i][j][k]);
            }
            dist2=drx1_2+dry2_2+drz1_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phiy[n][i][j][k]=phi_tanh(dist);
              dphiy[n][i][j][k]=dphi_tanh(dist);
              pvoly+=phiy[n][i][j][k];
              dpvoly+=fabs(dphiy[n][i][j][k]);
            }
            dist2=drx1_2+dry1_2+drz2_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phiz[n][i][j][k]=phi_tanh(dist);
              dphiz[n][i][j][k]=dphi_tanh(dist);
              pvolz+=phiz[n][i][j][k];
              dpvolz+=fabs(dphiz[n][i][j][k]);
            }
            dist2=drx2_2+dry2_2+drz1_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phixy[n][i][j][k]=phi_tanh(dist);
              pvolxy+=phixy[n][i][j][k];
            }
            dist2=drx1_2+dry2_2+drz2_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phiyz[n][i][j][k]=phi_tanh(dist);
              pvolyz+=phiyz[n][i][j][k];
            }
            dist2=drx2_2+dry1_2+drz2_2;
            if(dist2<radi2){
              double dist=sqrt(dist2)-radi;
              phizx[n][i][j][k]=phi_tanh(dist);
              pvolzx+=phizx[n][i][j][k];
            }
          }
        }
      }
      invpvc[n]=1.0/pvolc;
      invpvx[n]=1.0/pvolx;
      invpvy[n]=1.0/pvoly;
      invpvz[n]=1.0/pvolz;
      invpbvc[n]=1.0/dpvolc;
      invpbvx[n]=1.0/dpvolx;
      invpbvy[n]=1.0/dpvoly;
      invpbvz[n]=1.0/dpvolz;
  
  #pragma omp parallel for
      for(i=0;i<RD2;i++){
        int ii=RX+i-RD;
        if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
        for(int j=0;j<RD2;j++){
          int jj=RY+j-RD;
          if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
          for(int k=0;k<RD2;k++){
            int kk=RZ+k-RD;
            if(kk<=0){kk+=Nz;}else if(kk>Nz){kk-=Nz;}
            phit[ii][jj][kk]+=phic[n][i][j][k];
            phixyt[ii][jj][kk]+=phixy[n][i][j][k];
            phiyzt[ii][jj][kk]+=phiyz[n][i][j][k];
            phizxt[ii][jj][kk]+=phizx[n][i][j][k];
          }
        }
      }
    } 
  
    periodic_boundary(phit);
    periodic_boundary(phixyt);
    periodic_boundary(phiyzt);
    periodic_boundary(phizxt);
  }

void update_max_charge_density() {
    // This function calculates the charge density distribution assuming full dissociation (alpha=1)
    // It should be called after reading particle positions.

    // Reset cp_max and cp2_max
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                cp_max[i][j][k] = 0.0;
                cp2_max[i][j][k] = 0.0;
            }
        }
    }

    for (int n = 0; n < N; n++) {
        // Use N_sites instead of Nc
        double Nsn = fabs(N_sites[n]); 
        double coef1 = Nsn * invpbvc[n];
        int RX = (int)rx[n], RY = (int)ry[n], RZ = (int)rz[n];
        
        if (plabel[n] == 1) { // negative charge colloid
            for (int i = 0; i < RD2; i++) {
                int ii = RX + i - RD; if (ii <= 0) { ii += Nx; } else if (ii > Nx) { ii -= Nx; }
                for (int j = 0; j < RD2; j++) {
                    int jj = RY + j - RD; if (jj <= 0) { jj += Ny; } else if (jj > Ny) { jj -= Ny; }
                    for (int k = 0; k < RD2; k++) {
                        int kk = RZ + k - RD; if (kk <= 0) { kk += Nz; } else if (kk > Nz) { kk -= Nz; }
                        cp_max[ii][jj][kk] += coef1 * fabs(dphic[n][i][j][k]);
                    }
                }
            }
        } else { // positive charge colloid
            for (int i = 0; i < RD2; i++) {
                int ii = RX + i - RD; if (ii <= 0) { ii += Nx; } else if (ii > Nx) { ii -= Nx; }
                for (int j = 0; j < RD2; j++) {
                    int jj = RY + j - RD; if (jj <= 0) { jj += Ny; } else if (jj > Ny) { jj -= Ny; }
                    for (int k = 0; k < RD2; k++) {
                        int kk = RZ + k - RD; if (kk <= 0) { kk += Nz; } else if (kk > Nz) { kk -= Nz; }
                        cp2_max[ii][jj][kk] += coef1 * fabs(dphic[n][i][j][k]);
                    }
                }
            }
        }
    }
    periodic_boundary(cp_max);
    periodic_boundary(cp2_max);
}
/*************** integrated colloid surface charge, absolute value ***************/
void charge(){
  register int n,i;
  for(n=0;n<N;n++){
    double qn=0.0;
    //double coef1=0.5*Nc[i]*invpbvol[i]*invxi;
    double coef1=Nc[n]*invpbvc[n];
    double rxn=rx[n];
    double ryn=ry[n];
    double rzn=rz[n];
    
    int RX=(int)rxn;
    int RY=(int)ryn;
    int RZ=(int)rzn;
#pragma omp parallel for reduction(+:qn)
    for(i=0;i<RD2;i++){
      int ii=RX+i-RD;
      if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
      //double drx=(double)ii-rxn;if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
      for(int j=0;j<RD2;j++){
        int jj=RY+j-RD;
        if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
        //double dry=(double)jj-ryn;if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
        for(int k=0;k<RD2;k++){
          int kk=RZ+k-RD;
          if(kk<=0){kk+=Nz;}else if(kk>Nz){kk-=Nz;}
          qn+=coef1*fabs(dphic[n][i][j][k]);
        }
      }
    }
    if(plabel[n]==1)
      Z[n]=-qn;
    else
      Z[n]=qn;
  }
}

/*************** Update Nc[n] from local alpha(U) over each colloid ***************/
static inline void update_colloid_Nc_from_alpha() {
  for (int n = 0; n < N; ++n) {
    double qn = 0.0;
    const double coef = fabs(N_sites[n]) * invpbvc[n];
    const int RX = (int)rx[n], RY = (int)ry[n], RZ = (int)rz[n];
    for (int i = 0; i < RD2; ++i) {
      int ii = RX + i - RD; if (ii <= 0) { ii += Nx; } else if (ii > Nx) { ii -= Nx; }
      for (int j = 0; j < RD2; ++j) {
        int jj = RY + j - RD; if (jj <= 0) { jj += Ny; } else if (jj > Ny) { jj -= Ny; }
        for (int k = 0; k < RD2; ++k) {
          int kk = RZ + k - RD; if (kk <= 0) { kk += Nz; } else if (kk > Nz) { kk -= Nz; }
          const double Uijk = U[ii][jj][kk];
          double alpha_cell;
          if (plabel[n] == 1) { // Acid
            const double X = K_acid * exp(Uijk); 
            alpha_cell = X / (1.0 + X);
          } else { // Base
            const double X = K_base * exp(-Uijk); 
            alpha_cell = X / (1.0 + X);
          }
          qn += coef * alpha_cell * fabs(dphic[n][i][j][k]);
        }
      }
    }
    Nc[n] = qn;
  }
}
/***************update particle velocity***************/
void update_velocity_particle(){
  register int n,i;
  maxmin2(P);
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        P[i][j][k]-=ave;
      }
    }
  }
  periodic_boundary(P);
  
  maxmin2(vx);double ave_vx=ave;
  maxmin2(vy);double ave_vy=ave;
  maxmin2(vz);double ave_vz=ave;
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        vx[i][j][k]-=ave_vx;
        vy[i][j][k]-=ave_vy;
        vz[i][j][k]-=ave_vz;          
      }
    }
  }
  periodic_boundary(vx);
  periodic_boundary(vy);
  periodic_boundary(vz);
  
  
  //particle velocity
  for(n=0;n<N;n++){
    double invpvxn=invpvx[n];
    double invpvyn=invpvy[n];
    double invpvzn=invpvz[n];
    double rxn=rx[n];
    double ryn=ry[n];
    double rzn=rz[n];
    int RX=(int)rxn;
    int RY=(int)ryn;
    int RZ=(int)rzn;
    double vdx0=0.0,vdy0=0.0,vdz0=0.0;
#pragma omp parallel for reduction(+:vdx0,vdy0,vdz0)
    for(i=0;i<RD2;i++){
      int ii=RX+i-RD;
      double drx1=(double)ii-rxn;
      double drx2=drx1+0.5;
      double drx1_2=drx1*drx1;
      double drx2_2=drx2*drx2;
      if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
      for(int j=0;j<RD2;j++){
        int jj=RY+j-RD;
        double dry1=(double)jj-ryn;
        double dry2=dry1+0.5;
        double dry1_2=dry1*dry1;
        double dry2_2=dry2*dry2;
        if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
        for(int k=0;k<RD2;k++){
          int kk=RZ+k-RD;
          double drz1=(double)kk-rzn;
          double drz2=drz1+0.5;
          double drz1_2=drz1*drz1;
          double drz2_2=drz2*drz2;
          if(kk<=0){kk+=Nz;}else if(kk>Nz){kk-=Nz;}
          
          double vxijk=vx[ii][jj][kk];
          double vyijk=vy[ii][jj][kk];
          double vzijk=vz[ii][jj][kk];
          vdx0+=vxijk*phix[n][i][j][k];
          vdy0+=vyijk*phiy[n][i][j][k];
          vdz0+=vzijk*phiz[n][i][j][k];
        }
      }
    }
    vdx[n]=vdx0*invpvxn;
    vdy[n]=vdy0*invpvyn;
    vdz[n]=vdz0*invpvzn;
    //printf("%lf %lf %lf\n",vdx[n],vdy[n],vdz[n]);
  }
}
/***************MAC method***************/
void update_velocity_MAC(){
  register int i;
  double deta=etas-etal;
  double invDT=1.0/DT;
  double W = sqrt(2.*inv_beta*invDT); 
  // setup gaussian number generator
  const gsl_rng_type *T = (gsl_rng_type *)gsl_rng_default;
  gsl_rng **r = new gsl_rng*[NUMBER_THREADS];
  gsl_rng_env_setup();
  for(int m=0;m<NUMBER_THREADS;m++){
    r[m] = gsl_rng_alloc(T); 
    gsl_rng_set (r[m], (unsigned long int) std::time(NULL)+m);
  }
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    int threadNum = omp_get_thread_num();
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double vxijk=vx[i][j][k];
        double vyijk=vy[i][j][k];
        double vzijk=vz[i][j][k];
        double vxmcc=vx[i-1][j][k];
        double vxcpc=vx[i][j+1][k];
        double vxccp=vx[i][j][k+1];
        double vypcc=vy[i+1][j][k];
        double vycmc=vy[i][j-1][k];
        double vyccp=vy[i][j][k+1];
        double vzpcc=vz[i+1][j][k];
        double vzcpc=vz[i][j+1][k];
        double vzccm=vz[i][j][k-1];
        //viscous stress
        double eta1=etal+deta*phit[i][j][k];
        double etaxy=etal+deta*phixyt[i][j][k];
        double etayz=etal+deta*phiyzt[i][j][k];
        double etazx=etal+deta*phizxt[i][j][k];
        double vx_x=vxijk-vxmcc;
        double vx_y=vxcpc-vxijk;
        double vx_z=vxccp-vxijk;
        double vy_x=vypcc-vyijk;
        double vy_y=vyijk-vycmc;
        double vy_z=vyccp-vyijk;
        double vz_x=vzpcc-vzijk;
        double vz_y=vzcpc-vzijk;
        double vz_z=vzijk-vzccm;
        //Reynolds stress
        double vxc=0.5*(vxijk+vxmcc);
        double vyc=0.5*(vyijk+vycmc);
        double vzc=0.5*(vzijk+vzccm);
        double vxqxy=0.5*(vxijk+vxcpc);
        double vxqzx=0.5*(vxijk+vxccp);
        double vyqyz=0.5*(vyijk+vyccp);
        double vyqxy=0.5*(vyijk+vypcc);
        double vzqzx=0.5*(vzijk+vzpcc);
        double vzqyz=0.5*(vzijk+vzcpc);
        
        sigxx[i][j][k]=2.0*eta1*vx_x - Re*vxc*vxc;
        sigyy[i][j][k]=2.0*eta1*vy_y - Re*vyc*vyc;
        sigzz[i][j][k]=2.0*eta1*vz_z - Re*vzc*vzc;
        sigxy[i][j][k]=etaxy*(vx_y+vy_x) - Re*vxqxy*vyqxy;
        sigyz[i][j][k]=etayz*(vy_z+vz_y) - Re*vyqyz*vzqyz;
        sigzx[i][j][k]=etazx*(vz_x+vx_z) - Re*vzqzx*vxqzx;

        // random stress term
        if(noise==1){
          sigxx[i][j][k] -= sqrt(2.*eta1)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);
          sigyy[i][j][k] -= sqrt(2.*eta1)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);
          sigzz[i][j][k] -= sqrt(2.*eta1)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);
          sigxy[i][j][k] -= sqrt(etaxy)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);
          sigyz[i][j][k] -= sqrt(etayz)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);         
          sigzx[i][j][k] -= sqrt(etazx)*W*gsl_ran_gaussian_ziggurat(r[threadNum], 1);     
        }
      }
    }
  }
  
  periodic_boundary(sigxx);
  periodic_boundary(sigyy);
  periodic_boundary(sigzz);
  periodic_boundary(sigxy);
  periodic_boundary(sigyz);
  periodic_boundary(sigzx);
  //pressure: MAC method
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double t1=Fx[i][j][k]-Fx[i-1][j][k]+Fy[i][j][k]-Fy[i][j-1][k]+Fz[i][j][k]-Fz[i][j][k-1];
        double tv=invDT*(vx[i][j][k]-vx[i-1][j][k]+vy[i][j][k]-vy[i][j-1][k]+vz[i][j][k]-vz[i][j][k-1]);
        double ts=sigxx[i+1][j][k]+sigxx[i-1][j][k]-2.0*sigxx[i][j][k]
                 +sigyy[i][j+1][k]+sigyy[i][j-1][k]-2.0*sigyy[i][j][k]
                 +sigzz[i][j][k+1]+sigzz[i][j][k-1]-2.0*sigzz[i][j][k]
                 +2.0*(sigxy[i][j][k]+sigxy[i-1][j-1][k]-sigxy[i-1][j][k]-sigxy[i][j-1][k])
                 +2.0*(sigyz[i][j][k]+sigyz[i][j-1][k-1]-sigyz[i][j-1][k]-sigyz[i][j][k-1])
                 +2.0*(sigzx[i][j][k]+sigzx[i-1][j][k-1]-sigzx[i][j][k-1]-sigzx[i-1][j][k]);
        dP[i-1][j-1][k-1]=t1+ts+tv;
      }
    }
  }
  poisson(errorP,dP);
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int k=0;k<Nz;k++){
        P[i+1][j+1][k+1]=errorP[i][j][k];
      }
    }
  }
  periodic_boundary(P);
  
  
  
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double Pijk=P[i][j][k];
        double sigxyijk=sigxy[i][j][k];
        double sigyzijk=sigyz[i][j][k];
        double sigzxijk=sigzx[i][j][k];
        vxd[i][j][k]=-P[i+1][j][k]+Pijk+Fx[i][j][k]+sigxx[i+1][j][k]-sigxx[i][j][k]+sigxyijk-sigxy[i][j-1][k]+sigzxijk-sigzx[i][j][k-1];
        vyd[i][j][k]=-P[i][j+1][k]+Pijk+Fy[i][j][k]+sigyy[i][j+1][k]-sigyy[i][j][k]+sigyzijk-sigyz[i][j][k-1]+sigxyijk-sigxy[i-1][j][k];
        vzd[i][j][k]=-P[i][j][k+1]+Pijk+Fz[i][j][k]+sigzz[i][j][k+1]-sigzz[i][j][k]+sigzxijk-sigzx[i-1][j][k]+sigyzijk-sigyz[i][j-1][k];
      }
    }
  }
  maxmin2(vxd);double ave_vx=ave;
  maxmin2(vyd);double ave_vy=ave;
  maxmin2(vzd);double ave_vz=ave;
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double dvx=vxd[i][j][k]-ave_vx;
        double dvy=vyd[i][j][k]-ave_vy;
        double dvz=vzd[i][j][k]-ave_vz;
        vx[i][j][k]+=DT*dvx;
        vy[i][j][k]+=DT*dvy;
        vz[i][j][k]+=DT*dvz;
      }
    }
  }
  periodic_boundary(vx);
  periodic_boundary(vy);
  periodic_boundary(vz);
}
/*************** calculation of force ***************/
void calc_force(){//LJ
  register int i;
  double force_coef=-2.0*eM*wM/sig;
  double sigp=sig+2.0*xi;
  double sig22=sigp*sigp;
  double rthre2=4.0*sig22;
  double sig_morse=sig+2.0;
  Fene=0.0;
#pragma omp parallel for
  for(i=0;i<N*numid;i++){Fdxid[i]=0.0;Fdyid[i]=0.0;Fdzid[i]=0.0;}
#pragma omp parallel for reduction(+:Fene) schedule(dynamic,DNUM)
  for(i=0;i<N;i++){
    int myid=omp_get_thread_num();
    double rxi=rx[i],ryi=ry[i],rzi=rz[i];
    double fxi=0.0,fyi=0.0,fzi=0.0;
    for(int k=1;k<=connect[i][0];k++){
      int j=connect[i][k];// j>i (see update)
      double r_ij_x=rxi-rx[j];if(r_ij_x>HLx){r_ij_x-=Lx;}else if(r_ij_x<-HLx){r_ij_x+=Lx;}
      double r_ij_y=ryi-ry[j];if(r_ij_y>HLy){r_ij_y-=Ly;}else if(r_ij_y<-HLy){r_ij_y+=Ly;}
      double r_ij_z=rzi-rz[j];if(r_ij_z>HLz){r_ij_z-=Lz;}else if(r_ij_z<-HLz){r_ij_z+=Lz;}
      double r_ij2=r_ij_x*r_ij_x+r_ij_y*r_ij_y+r_ij_z*r_ij_z;
      if(r_ij2 > r02){continue;}
      //soft core
      double invr2=1.0/r_ij2;
      double r2=sig22*invr2;
      double r6=r2*r2*r2;
      double r12=r6*r6;
      double coef=12.0*scr*r12*invr2;
      Fene+=scr*r12;
      double forcex=coef*r_ij_x;
      double forcey=coef*r_ij_y;
      double forcez=coef*r_ij_z;
      fxi+=forcex;
      fyi+=forcey;
      fzi+=forcez;
      int l=myid*N+j;
      Fdxid[l]-=forcex;
      Fdyid[l]-=forcey;
      Fdzid[l]-=forcez;
    }
    Fdx[i]+=fxi;
    Fdy[i]+=fyi;
    Fdz[i]+=fzi;
  }
  
  for(int myid=0;myid<numid;myid++){
#pragma omp parallel for
    for(i=0;i<N;i++){
      int k=myid*N+i;
      Fdx[i]+=Fdxid[k];
      Fdy[i]+=Fdyid[k];
      Fdz[i]+=Fdzid[k];
    }
  }
}
/*************** region partitioning method ***************/
void Cellupdate(){
  register int i;
  int imax=0.0;
#pragma omp parallel for reduction(max:imax) schedule(dynamic,DNUM)
  for(i=0;i<N;i++){
    int count=0;
    double rxi=rx[i],ryi=ry[i],rzi=rz[i];
    rx_old[i]=rx[i];ry_old[i]=ry[i];rz_old[i]=rz[i];
    for(int j=i+1;j<N;j++){
      double r_ij_x=rxi-rx[j];if(r_ij_x>HLx){r_ij_x-=Lx;}else if(r_ij_x<-HLx){r_ij_x+=Lx;}
      double r_ij_y=ryi-ry[j];if(r_ij_y>HLy){r_ij_y-=Ly;}else if(r_ij_y<-HLy){r_ij_y+=Ly;}
      double r_ij_z=rzi-rz[j];if(r_ij_z>HLz){r_ij_z-=Lz;}else if(r_ij_z<-HLz){r_ij_z+=Lz;}
      double r_ij2=r_ij_x*r_ij_x+r_ij_y*r_ij_y+r_ij_z*r_ij_z;
      if(r_ij2<r1_2){count++;connect[i][count]=j;}
    }
    connect[i][0]=count;
    if(count>imax){imax=count;}
  }
  //printf("cellupdate t=%.3f adj_max=%d\n",t,imax);
  if(imax>=adj){printf("adj is too small!\n");exit(1);}
}
/*************** periodic boundary ***************/
void periodic_boundary(double A[NX][NY][NZ]){
  register int i;
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      A[i][j][0]=A[i][j][Nz];
      A[i][j][Nz+1]=A[i][j][1];
    }
  }
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=0;j<=Nz+1;j++){
      A[i][0][j]=A[i][Ny][j];
      A[i][Ny+1][j]=A[i][1][j];
    }
  }
#pragma omp parallel for
  for(i=0;i<=Ny+1;i++){
    for(int j=0;j<=Nz+1;j++){
      A[0][i][j]=A[Nx][i][j];
      A[Nx+1][i][j]=A[1][i][j];
    }
  }
}

/*************** Laplacian ***************/
inline double Laplacian1(int i,int j,int k,double A[NX][NY][NZ]){
  double lap=A[i-1][j][k]+A[i][j-1][k]+A[i][j][k-1]+A[i][j][k+1]+A[i][j+1][k]+A[i+1][j][k]-6.0*A[i][j][k];
  return lap;
}

inline double Laplacian2(int i,int j,int k,double A[NX][NY][NZ]){
  double t1=A[i-1][j][k]+A[i][j-1][k]+A[i][j][k-1]+A[i][j][k+1]+A[i][j+1][k]+A[i+1][j][k];
  double t2=A[i-1][j-1][k-1]+A[i-1][j-1][k+1]+A[i-1][j+1][k-1]+A[i-1][j+1][k+1]
           +A[i+1][j-1][k-1]+A[i+1][j-1][k+1]+A[i+1][j+1][k-1]+A[i+1][j+1][k+1];
  double lap=0.5*(t1+0.5*t2-10.0*A[i][j][k]);
  return lap;
}
inline double Divgrad1(int i,int j,int k,double A[NX][NY][NZ],double B[NX][NY][NZ]){
  double Aijk=A[i][j][k];
  double Bijk=B[i][j][k];
  double divgrad=(A[i-1][j][k]+Aijk)*(B[i-1][j][k]-Bijk)
                +(A[i][j-1][k]+Aijk)*(B[i][j-1][k]-Bijk)
                +(A[i][j][k-1]+Aijk)*(B[i][j][k-1]-Bijk)
                +(A[i][j][k+1]+Aijk)*(B[i][j][k+1]-Bijk)
                +(A[i][j+1][k]+Aijk)*(B[i][j+1][k]-Bijk)
                +(A[i+1][j][k]+Aijk)*(B[i+1][j][k]-Bijk);
  divgrad*=0.5;
  return divgrad;
}
inline double Divgrad2(int i,int j,int k,double A[NX][NY][NZ],double B[NX][NY][NZ]){
  double Aijk=A[i][j][k];
  double Bijk=B[i][j][k];
  double t1=(A[i-1][j][k]+Aijk)*(B[i-1][j][k]-Bijk)
           +(A[i][j-1][k]+Aijk)*(B[i][j-1][k]-Bijk)
           +(A[i][j][k-1]+Aijk)*(B[i][j][k-1]-Bijk)
           +(A[i][j][k+1]+Aijk)*(B[i][j][k+1]-Bijk)
           +(A[i][j+1][k]+Aijk)*(B[i][j+1][k]-Bijk)
           +(A[i+1][j][k]+Aijk)*(B[i+1][j][k]-Bijk);
  double t2=(A[i-1][j-1][k-1]+Aijk)*(B[i-1][j-1][k-1]-Bijk)
           +(A[i-1][j-1][k+1]+Aijk)*(B[i-1][j-1][k+1]-Bijk)
           +(A[i-1][j+1][k-1]+Aijk)*(B[i-1][j+1][k-1]-Bijk)
           +(A[i-1][j+1][k+1]+Aijk)*(B[i-1][j+1][k+1]-Bijk)
           +(A[i+1][j-1][k-1]+Aijk)*(B[i+1][j-1][k-1]-Bijk)
           +(A[i+1][j-1][k+1]+Aijk)*(B[i+1][j-1][k+1]-Bijk)
           +(A[i+1][j+1][k-1]+Aijk)*(B[i+1][j+1][k-1]-Bijk)
           +(A[i+1][j+1][k+1]+Aijk)*(B[i+1][j+1][k+1]-Bijk);
  double divgrad=0.25*(t1+0.5*t2);
  return divgrad;
}
/*************** update electric potential***************/
//use either depending on your demand
/*************** counterion distribution NOT updated -- see commented lines in the main routine for their update ***************/
void poisson(double A[Nx][Ny][Nz],double B[Nx][Ny][Nz]){
  register int i;
  fftw_plan plan1;
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int k=0;k<Nz;k++){
        fftw_in[i*Ny*Nz+j*Nz+k]=B[i][j][k];
      }
    }
  }
  plan1=fftw_plan_dft_r2c_3d(Nx,Ny,Nz,fftw_in,fftw_out,FFTW_ESTIMATE);
  fftw_execute(plan1);
  fftw_destroy_plan(plan1);
  double dphasex=PI2/(double)Nx;
  double dphasey=PI2/(double)Ny;
  double dphasez=PI2/(double)Nz;
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    double cosx=cos(dphasex*(double)i);
    for(int j=0;j<Ny;j++){
      double cosy=cos(dphasey*(double)j);
      for(int k=0;k<=HNz;k++){
        if(i==0 && j==0 && k==0){continue;}
        double cosz=cos(dphasez*(double)k);
        double cos2=cosx+cosy+cosz-3.0;
        double norm=0.5/cos2;//only nearest neighbor
        fftw_out[i*Ny*Nz2+j*Nz2+k][0]*=norm;
        fftw_out[i*Ny*Nz2+j*Nz2+k][1]*=norm;
      }
    }
  }
  fftw_out[0][0]=0.0;
  fftw_out[0][1]=0.0;
  plan1=fftw_plan_dft_c2r_3d(Nx,Ny,Nz,fftw_out,fftw_in,FFTW_ESTIMATE);
  fftw_execute(plan1);
  fftw_destroy_plan(plan1);
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int k=0;k<Nz;k++){
        A[i][j][k]=invVol*fftw_in[i*Ny*Nz+j*Nz+k];
      }
    }
  }
}
/*************** counterion distribution updated ***************/
void update_electric_equilibrium(){
  register int i;
  max=1.0;
  int ite=0;
  int ite_max=10000;
  double ethre=1.0e-4;
  double sor_coef=1.5;//determined empirically
  while(max>ethre || min<-ethre){
    max=-100.0,min=100.0;
#pragma omp parallel for reduction(max:max) reduction(min:min)
    for(i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          double rho0=-invep*rho[i][j][k];
          //only nearest
          double tt1=U[i-1][j][k]+U[i][j-1][k]+U[i][j][k-1]+U[i][j][k+1]+U[i][j+1][k]+U[i+1][j][k];
          double source=tt1 - rho0;
          double errU=source/6.0-U[i][j][k];
          //include diagonal
          //double tt1=U[i-1][j][k]+U[i][j-1][k]+U[i][j][k-1]+U[i][j][k+1]+U[i][j+1][k]+U[i+1][j][k];
          //double tt2=U[i-1][j-1][k-1]+U[i-1][j-1][k+1]+U[i-1][j+1][k-1]+U[i-1][j+1][k+1]
          //          +U[i+1][j-1][k-1]+U[i+1][j-1][k+1]+U[i+1][j+1][k-1]+U[i+1][j+1][k+1];
          //double source=0.5*(tt1+0.25*tt2) - rho0;
          //double errU=0.25*source-U[i][j][k];
          
          if(max<errU){max=errU;}
          if(min>errU){min=errU;}
          errU*=sor_coef;
          if(errU>0.01){errU=0.01;}
          else if(errU<-0.01){errU=-0.01;}
          U[i][j][k]+=errU;
        }
      }
    }
    periodic_boundary(U);
    
    // MODIFIED: Use K_acid and K_base correctly for each particle type.
    #pragma omp parallel for
    for(i=1;i<=Nx;i++){
        for(int j=1;j<=Ny;j++){
            for(int k=1;k<=Nz;k++){
                double Uijk = U[i][j][k];

                // For acidic groups (plabel 1 -> negative charge)
                // dissociation increases with potential: exp(+U)
                double X_acid = K_acid * exp(Uijk);
                double alpha_acid = X_acid / (1.0 + X_acid);
                cp[i][j][k] = cp_max[i][j][k] * alpha_acid;
                
                // For basic groups (plabel 2 -> positive charge)
                // dissociation increases with potential: exp(-U)
                double X_base = K_base * exp(-Uijk);
                double alpha_base = X_base / (1.0 + X_base);
                cp2[i][j][k] = cp2_max[i][j][k] * alpha_base;
            }
        }
    }
                

    double cpt=0.0,cct=0.0;
    double cp2t=0.0,cc2t=0.0;
    double cspt=0.0,csmt=0.0;
#pragma omp parallel for reduction(+:cpt,cct,cp2t,cc2t,cspt,csmt)
    for(i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          cpt+=cp[i][j][k];
          cc[i][j][k]=invscale3*exp(-U[i][j][k]-gc*phit[i][j][k]);
          cct+=cc[i][j][k];
          //polyion2 (cation)
          cp2t+=cp2[i][j][k];
          cc2[i][j][k]=invscale3*exp(U[i][j][k]-gc*phit[i][j][k]);
          cc2t+=cc2[i][j][k];
          //salt
          csp[i][j][k]=exp(-U[i][j][k]-gc*phit[i][j][k]);
          cspt+=csp[i][j][k];
          csm[i][j][k]=exp(U[i][j][k]-gc*phit[i][j][k]);
          csmt+=csm[i][j][k];
        }
      }
    }
    double normcc = (cpt > 1e-9 && cct > 1e-9) ? cpt/cct : 0.0;
    double normcc2 = (cp2t > 1e-9 && cc2t > 1e-9) ? cp2t/cc2t : 0.0;

    double normcsp = (cspt > 1e-9) ? (n0*Volume)/cspt : 0.0;
    double normcsm = (csmt > 1e-9) ? (n0*Volume)/csmt : 0.0;
#pragma omp parallel for
    for(i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          cc[i][j][k]*=normcc;
          cc2[i][j][k]*=normcc2;
          csp[i][j][k]*=normcsp;
          csm[i][j][k]*=normcsm;
          rho[i][j][k]=-cp[i][j][k]+cc[i][j][k]+cp2[i][j][k]-cc2[i][j][k]+csp[i][j][k]-csm[i][j][k];
        }
      }
    }
    ite++;
    if(ite>ite_max){break;}
  }
  periodic_boundary(U);
  periodic_boundary(cp);
  periodic_boundary(cp2);
  periodic_boundary(cc);
  periodic_boundary(cc2);
  periodic_boundary(csp);
  periodic_boundary(csm);
  if(int(t/DT)%interval_pos==0)
    std::cout <<"MD steps:"<<int(t/DT)<<' '<<"iteration steps:"<<ite<<' '<<"maximum error in U:"<<max<<std::endl;
}
void initial(int){
  fr=fopen(filename,"r"); 
  for(int l=0;l<N;l++)
    fscanf(fr,"%d %lf %lf %lf %lf",&plabel[l],&rx[l],&ry[l],&rz[l],&N_sites[l]);
  fclose (fr);
  
    for(int i=0;i<N;i++){
      if(abs(plabel[i]-1)>0&&abs(plabel[i]-2)>0){
        std::cout <<"error: wrong type IDs"<<std::endl;
        exit(0);
      }
    }
    #pragma omp master
    min=100;
    for(int i=0;i<N;i++){
      double rxi=rx[i],ryi=ry[i],rzi=rz[i];
      for(int j=i+1;j<N;j++){
        double r_ij_x=rxi-rx[j];if(r_ij_x>HLx){r_ij_x-=Lx;}else if(r_ij_x<-HLx){r_ij_x+=Lx;}
        double r_ij_y=ryi-ry[j];if(r_ij_y>HLy){r_ij_y-=Ly;}else if(r_ij_y<-HLy){r_ij_y+=Ly;}
        double r_ij_z=rzi-rz[j];if(r_ij_z>HLz){r_ij_z-=Lz;}else if(r_ij_z<-HLz){r_ij_z+=Lz;}
        double r_ij2=r_ij_x*r_ij_x+r_ij_y*r_ij_y+r_ij_z*r_ij_z;
        if(min > sqrt(r_ij2)) { min = sqrt(r_ij2); }
      }
    }
  

    update_shape_function();
    update_max_charge_density();
  
    // Simple initial guess for counter-ion concentration
    double initial_alpha_acid = K_acid / (1.0 + K_acid);
    double initial_alpha_base = K_base / (1.0 + K_base);
    double total_charge_guess1 = 0.0; 
    double total_charge_guess2 = 0.0; 
  
    for(int l=0; l<N; ++l) {
      if(plabel[l] == 1) { // Acid
          total_charge_guess1 -= N_sites[l] * initial_alpha_acid;
      } else { // Base
          total_charge_guess2 += N_sites[l] * initial_alpha_base; 
      }
    }
  

    for(int i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          cc[i][j][k] = fabs(total_charge_guess1) / Volume;
          cc2[i][j][k] = fabs(total_charge_guess2) / Volume;
          csp[i][j][k] = n0;
          csm[i][j][k] = n0;
        }
      }
    }
 

  #pragma omp parallel for
    for(int i=1;i<=Nx;i++){
      for(int j=1;j<=Ny;j++){
        for(int k=1;k<=Nz;k++){
          rho[i][j][k]=-cp[i][j][k]+cc[i][j][k]+cp2[i][j][k]-cc2[i][j][k]+csp[i][j][k]-csm[i][j][k];
        }
      }
    }
    
    periodic_boundary(cp);
    periodic_boundary(cp2);
    periodic_boundary(cc);
    periodic_boundary(cc2);
    periodic_boundary(rho);
    periodic_boundary(csp);
    periodic_boundary(csm);
  
    update_electric_equilibrium();
  }
void write_data(){
  fprintf(fw,"ITEM: TIMESTEP\n"); 
  fprintf(fw,"%d\n",int(t/DT)); 
  fprintf(fw,"ITEM: NUMBER OF ATOMS\n"); 
  fprintf(fw,"%d\n",N);
  fprintf(fw,"ITEM: BOX BOUNDS pp pp pp\n"); 
  fprintf(fw,"0 %d\n",Nx);
  fprintf(fw,"0 %d\n",Ny);
  fprintf(fw,"0 %d\n",Nz);
  fprintf(fw,"ITEM: ATOMS id type q x y z radius\n"); 
  for(int i=0;i<N;i++){
    fprintf(fw,"%d %d %lf %lf %lf %lf %lf\n",(i+1),plabel[i],Z[i],rx[i],ry[i],rz[i],RADI); 
    }
}
void write_potential(){
  fprintf(fp,"ITEM: TIMESTEP\n"); 
  fprintf(fp,"%d\n",int(t/DT)); 
  for(int i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        fprintf(fp,"%lf\n",U[i][j][k]);
      }
    }
  }
}
void write_rho(){
  fprintf(frho,"ITEM: TIMESTEP\n"); 
  fprintf(frho,"%d\n",int(t/DT)); 
  for(int i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        fprintf(frho,"%lf\n",rho[i][j][k]);
      }
    }
  }
}
void write_vel(){
  fprintf(fvel,"ITEM: TIMESTEP\n"); 
  fprintf(fvel,"%d\n",int(t/DT)); 
  for(int i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        fprintf(fvel,"%lf %lf %lf\n",vx[i][j][k],vy[i][j][k],vz[i][j][k]);
      }
    }
  }
}
void calc_lattice(void){
  register int n,i;
  
  //calculate chemical potential and force due to ionic term
#pragma omp parallel for
  for(i=0;i<=Nx+1;i++){
    double exE1=-exE*(double)i;
    for(int j=0;j<=Ny+1;j++){
      for(int k=0;k<=Nz+1;k++){
        double Uijk=U[i][j][k] + exE1;
        double phitijk=phit[i][j][k];
        double ccijk=cc[i][j][k];
        double logcc=log(ccijk);
        //muc[i][j][k]=Uijk+logcc+gc*phitijk;
        double cc2ijk=cc2[i][j][k];
        double logcc2=log(cc2ijk);
        //muc2[i][j][k]=-Uijk+logcc2+gc*phitijk;
        double cspijk=csp[i][j][k];
        double logcspijk=log(cspijk);
        double csmijk=csm[i][j][k];
        double logcsmijk=log(csmijk);
        rhot[i][j][k]=ccijk+cc2ijk+cspijk+csmijk;
        rhop[i][j][k]=ccijk-cc2ijk+cspijk-csmijk;
      }
    }
  }
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        double Uijk=U[i][j][k];
        double phitijk=phit[i][j][k];
        double rhotijk=rhot[i][j][k];
        double rhotpcc=rhot[i+1][j][k];
        double rhotcpc=rhot[i][j+1][k];
        double rhotccp=rhot[i][j][k+1];
        double rhopijk=rhop[i][j][k];
        //common g: gc=gc2
        Fx[i][j][k]=-0.5*(rhopijk+rhop[i+1][j][k])*(U[i+1][j][k]-Uijk-exE) - (rhotpcc-rhotijk) - 0.5*gc*(rhotpcc+rhotijk)*(phit[i+1][j][k]-phitijk);
        Fy[i][j][k]=-0.5*(rhopijk+rhop[i][j+1][k])*(U[i][j+1][k]-Uijk    ) - (rhotcpc-rhotijk) - 0.5*gc*(rhotcpc+rhotijk)*(phit[i][j+1][k]-phitijk);
        Fz[i][j][k]=-0.5*(rhopijk+rhop[i][j][k+1])*(U[i][j][k+1]-Uijk    ) - (rhotccp-rhotijk) - 0.5*gc*(rhotccp+rhotijk)*(phit[i][j][k+1]-phitijk);
        //dcc[i][j][k]=(advection(i,j,k,cc)) + Dc*(Divgrad2(i,j,k,cc,muc));
        //dcc2[i][j][k]=(advection(i,j,k,cc2)) + Dc2*(Divgrad2(i,j,k,cc2,muc2));
      }
    }
  }
  
  //direct interparitcle force
#pragma omp parallel for
  for(i=0;i<N;i++){Fdx[i]=0.0;Fdy[i]=0.0;Fdz[i]=0.0;}
  max=0.2*(r1-r0)*(r1-r0);
#pragma omp parallel for reduction(max:max)
  for(i=0;i<N;i++){
    double drx=rx[i]-rx_old[i];if(drx<-HLx){drx+=Lx;}else if(drx>HLx){drx-=Lx;}
    double dry=ry[i]-ry_old[i];if(dry<-HLy){dry+=Ly;}else if(dry>HLy){dry-=Ly;}
    double drz=rz[i]-rz_old[i];if(drz<-HLz){drz+=Lz;}else if(drz>HLz){drz-=Lz;}
    double r_ij2=drx*drx+dry*dry+drz*drz;
    if(r_ij2>max){max=r_ij2;}
  }
  double thre=0.25*(r1-r0)*(r1-r0);
  if(max>thre){Cellupdate();}
  calc_force();
  
  
  
  //calculate reversible stress and update particle positions
  for(n=0;n<N;n++){
    double Ncn=Nc[n];
    double coefx=Ncn*invpbvx[n];//positive Nc means negative charge
    double coefy=Ncn*invpbvy[n];
    double coefz=Ncn*invpbvz[n];
    if(plabel[n]==2){
      coefx=-coefx;coefy=-coefy;coefz=-coefz;
    }
    double rxn=rx[n];
    double ryn=ry[n];
    double rzn=rz[n];
    int RX=(int)rxn;
    int RY=(int)ryn;
    int RZ=(int)rzn;
    double Fxn=Fdx[n];
    double Fyn=Fdy[n];
    double Fzn=Fdz[n];
#pragma omp parallel for reduction(+:Fxn,Fyn,Fzn)
    for(i=0;i<RD2;i++){
      int ii=RX+i-RD;
      double drx1=(double)ii-rxn;
      double drx2=drx1+0.5;
      double drx1_2=drx1*drx1;
      double drx2_2=drx2*drx2;
      if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
      for(int j=0;j<RD2;j++){
        int jj=RY+j-RD;
        double dry1=(double)jj-ryn;
        double dry2=dry1+0.5;
        double dry1_2=dry1*dry1;
        double dry2_2=dry2*dry2;
        if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
        for(int k=0;k<RD2;k++){
          int kk=RZ+k-RD;
          double drz1=(double)kk-rzn;
          double drz2=drz1+0.5;
          double drz1_2=drz1*drz1;
          double drz2_2=drz2*drz2;
          if(kk<=0){kk+=Nz;}else if(kk>Nz){kk-=Nz;}
          
          double Uijk=U[ii][jj][kk];
          double rhotijk=rhot[ii][jj][kk];
          
          //Fx
          double dist2=drx2_2+dry1_2+drz1_2;
          if(dist2<radi2 && dist2>0.0001){
            double dist=sqrt(dist2);
            double dspx=dphix[n][i][j][k];
            double dpx=dspx*drx2/dist;//d phi / d x
            double Ex=Uijk-U[ii+1][jj][kk] + exE;
            Fxn+=0.5*gc*(rhotijk+rhot[ii+1][jj][kk])*dpx;
            Fxn+=-coefx*fabs(dspx)*Ex;//minus sign is needed since the sign of Ncn is reversed
          }
          //Fy
          dist2=drx1_2+dry2_2+drz1_2;
          if(dist2<radi2 && dist2>0.0001){
            double dist=sqrt(dist2);
            double dspy=dphiy[n][i][j][k];
            double dpy=dspy*dry2/dist;
            double Ey=Uijk-U[ii][jj+1][kk];
            Fyn+=0.5*gc*(rhotijk+rhot[ii][jj+1][kk])*dpy;
            Fyn+=-coefy*fabs(dspy)*Ey;
          }
          //Fz
          dist2=drx1_2+dry1_2+drz2_2;
          if(dist2<radi2 && dist2>0.0001){
            double dist=sqrt(dist2);
            double dspz=dphiz[n][i][j][k];
            double dpz=dspz*drz2/dist;
            double Ez=Uijk-U[ii][jj][kk+1];
            Fzn+=0.5*gc*(rhotijk+rhot[ii][jj][kk+1])*dpz;
            Fzn+=-coefz*fabs(dspz)*Ez;
          }
        }
      }
    }
    Fdx[n]=Fxn;
    Fdy[n]=Fyn;
    Fdz[n]=Fzn;
    //F: direct force
    Fxn*=invpvx[n];
    Fyn*=invpvy[n];
    Fzn*=invpvz[n];
    //if(branch==-2){Fxn=0.0;Fyn=0.0;Fzn=0.0;}
    //include direct interaction to fluid force, and update particle coordinate
#pragma omp parallel for
    for(i=0;i<RD2;i++){
      int ii=RX+i-RD;
      if(ii<=0){ii+=Nx;}else if(ii>Nx){ii-=Nx;}
      for(int j=0;j<RD2;j++){
        int jj=RY+j-RD;
        if(jj<=0){jj+=Ny;}else if(jj>Ny){jj-=Ny;}
        for(int k=0;k<RD2;k++){
          int kk=RZ+k-RD;
          if(kk<=0){kk+=Nz;}else if(kk>Nz){kk-=Nz;}
          Fx[ii][jj][kk]+=Fxn*phix[n][i][j][k];
          Fy[ii][jj][kk]+=Fyn*phiy[n][i][j][k];
          Fz[ii][jj][kk]+=Fzn*phiz[n][i][j][k];
        }
      }
    }
  }
  periodic_boundary(Fx);
  periodic_boundary(Fy);
  periodic_boundary(Fz);
  
  /*
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        cc[i][j][k]+=DT*dcc[i][j][k];
        cc2[i][j][k]+=DT*dcc2[i][j][k];
      }
    }
  }
  */
  update_velocity_MAC();
  update_velocity_particle();
#pragma omp parallel for
  for(n=0;n<N;n++){
    double rxn=rx[n]+DT*vdx[n];
    double ryn=ry[n]+DT*vdy[n];
    double rzn=rz[n]+DT*vdz[n];
    if(rxn>=Lx){rxn-=Lx;}else if(rxn<0.0){rxn+=Lx;}
    if(ryn>=Ly){ryn-=Ly;}else if(ryn<0.0){ryn+=Ly;}
    if(rzn>=Lz){rzn-=Lz;}else if(rzn<0.0){rzn+=Lz;}
    rx[n]=rxn;
    ry[n]=ryn;
    rz[n]=rzn;
  }
  update_shape_function();
  update_electric_equilibrium();
  update_colloid_Nc_from_alpha(); // Update colloid total charge based on local alpha
  //update charge density
#pragma omp parallel for
  for(i=1;i<=Nx;i++){
    for(int j=1;j<=Ny;j++){
      for(int k=1;k<=Nz;k++){
        rho[i][j][k]=-cp[i][j][k]+cc[i][j][k]+cp2[i][j][k]-cc2[i][j][k]+csp[i][j][k]-csm[i][j][k];
      }
    }
  }
  periodic_boundary(cp);
  periodic_boundary(cp2);
  periodic_boundary(cc);
  periodic_boundary(cc2);
  periodic_boundary(rho);
  periodic_boundary(csp);
  periodic_boundary(csm);
  
  //update electrostatic potential by solving Poisson equation
  //update_electric_field();
  if(int(t/DT)%interval_pos==0){
    charge();
    write_data();
    fflush(fw);
  }
  if(int(t/DT)%interval_field==0){
    write_potential();
    write_rho();
    write_vel();
    fflush(fp);
    fflush(frho);
    fflush(fvel);
  }
  
  if(t>quit_time){exit(0);}
  t+=DT;step++;
}/******************end of calc lattice********************/
/*************************main***********************/
int main(int argc, char **argv){
  std::time_t timer; std::time(&timer);
  std::cout.precision(15);
  fw=fopen("dump_fpd.dat","w"); 
  fp=fopen("potential.dat","w");  
  frho=fopen("rho.dat","w");
  fvel=fopen("vel.dat","w");
  register int i;
#ifdef _OPENMP
  numid=omp_get_max_threads();
  omp_set_num_threads(NUMBER_THREADS);
  fftw_plan_with_nthreads(numid);
#endif
  if(NULL==(fftw_in =(double*)fftw_malloc(sizeof(double)*Nx*Ny*2*Nz2))){(void)printf("fftw malloc error\n");exit(1);}
  if(NULL==(fftw_out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*Nx*Ny*Nz2))){(void)printf("fftw malloc error\n");exit(1);}
  
  if(NULL==(Fdxid=new(std::nothrow) double[numid*N])){(void)printf("new error_Fdx\n");exit(1);}
  if(NULL==(Fdyid=new(std::nothrow) double[numid*N])){(void)printf("new error_Fdy\n");exit(1);}
  if(NULL==(Fdzid=new(std::nothrow) double[numid*N])){(void)printf("new error_Fdz\n");exit(1);}
  
  // MODIFIED: Check for the correct number of arguments (program name + 8 parameters = 9)
  if(argc!=9){
    // MODIFIED: Update usage message to include pKa, pKb and pH
    std::cout << "Usage: " << argv[0] <<" [init] [time] [beta] [eps] [c_salt] [pKa] [pKb] [pH]" << std::endl;
    exit(0);
  }

  // setting parameters
  filename=argv[1];
  quit_time=atof(argv[2]);
  beta=atof(argv[3]);
  inv_beta=1.0/beta;
  scr=atof(argv[4]);
  n0=atof(argv[5]);
  double pKa=atof(argv[6]); // Read pKa for acid
  double pKb=atof(argv[7]); // Read pKb for base
  double pH=atof(argv[8]);  // Read pH

  // Calculate the correct constants for acid and base separately
  K_acid = pow(10.0, pH - pKa);
  double pOH = 14.0 - pH;
  K_base = pow(10.0, pOH - pKb);

  // output setting parameters
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "INFO: Running CHARGE REGULATION (CR) simulation." << std::endl;
  std::cout << "pKa=" << pKa << ", pKb=" << pKb << ", pH=" << pH << std::endl;
  std::cout << "Calculated K_acid = " << K_acid << ", K_base = " << K_base << std::endl;
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "init state: " << filename << std::endl;
  std::cout << "time range: " << quit_time << std::endl; 
  std::cout << "beta=" << beta <<", eps(LJ6)="<< scr << ", c_salt="<< n0<<std::endl;
  std::cout << "radi=" << RADI << ", Ncol="<<(Ncp+Ncp2)<<std::endl;
  std::cout << "box=" << Nx << ", vol="<<double((Ncp+Ncp2)*(4.0/3*PI*RADI*RADI*RADI)/(Nx*Ny*Nz))<<std::endl;
  if(noise==0)
      std::cout << "thermal noise off" <<std::endl; 
  else
      std::cout << "thermal noise on" <<std::endl;
  

#pragma omp parallel for
  for(i=0;i<N*numid;i++){
    Fdxid[i]=0.0;Fdyid[i]=0.0;Fdzid[i]=0.0;
  }
  
#pragma omp master
  for(i=0;i<N;i++){
    rx[i]=0.0;ry[i]=0.0;rz[i]=0.0;
    vdx[i]=0.0,vdy[i]=0.0;vdz[i]=0.0;
    Fdx[i]=0.0;Fdy[i]=0.0;Fdz[i]=0.0;
    rx_old[i]=0.0;ry_old[i]=0.0;rz_old[i]=0.0;
    Z[i]=0.0;
    //Nd[i]=Nd1;
    Nc[i]=0.0;
  }
#pragma omp parallel for
  for(i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        phit[i][j][k]=0.0;phixyt[i][j][k]=0.0;phiyzt[i][j][k]=0.0;phizxt[i][j][k]=0.0;
        vx[i][j][k]=0.0;vy[i][j][k]=0.0;vz[i][j][k]=0.0;
        Fx[i][j][k]=0.0;Fy[i][j][k]=0.0;Fz[i][j][k]=0.0;
        P[i][j][k]=0.0;
        sigxx[i][j][k]=0.0;sigyy[i][j][k]=0.0;sigzz[i][j][k]=0.0;
        sigxy[i][j][k]=0.0;sigyz[i][j][k]=0.0;sigzx[i][j][k]=0.0;
        
        //eloectrostatic
        cp[i][j][k]=0.0,cc[i][j][k]=0.0;
        U[i][j][k]=0.0;
        rho[i][j][k]=0.0;rhot[i][j][k]=0.0;rhop[i][j][k]=0.0;
        muc[i][j][k]=0.0;mu1[i][j][k]=0.0;mu2[i][j][k]=0.0;
        dcc[i][j][k]=0.0;
        
        cp2[i][j][k]=0.0;cc2[i][j][k]=0.0;
        csp[i][j][k]=n0;csm[i][j][k]=n0;
      }
    }
  }
#pragma omp parallel for
  for(i=0;i<Nx;i++){
    for(int j=0;j<Ny;j++){
      for(int k=0;k<Nz;k++){
        dP[i][j][k]=0.0,errorP[i][j][k]=0.0;
        dU[i][j][k]=0.0,errorU[i][j][k]=0.0;
      }
    }
  }
  initial(0);
  while(t<quit_time){calc_lattice();}
  delete [] Fdxid;
  delete [] Fdyid;
  delete [] Fdzid;
  fftw_free(fftw_in);
  fftw_free(fftw_out);
  fflush(fw);fflush(fp);fflush(frho);
  fclose(fw);fclose(fp);fclose(frho);
  return 0;
}