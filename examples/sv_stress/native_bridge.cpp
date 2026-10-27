#include "hapc_core.hpp"
#include <cstring>
extern "C" int fit(int logistic,int n,int k,int q,const double* y,const double* z,const double* v,const double* init,int steps,double tol,double factor,double* history,double* risk,char* status){
 try {
 Eigen::Map<const VectorXd> Y(y,n), A(init,k);
 Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> Z(z,n,k),V(v,q,k);
#ifdef REPORT_STATUS
 auto out=logistic ? pcghal_boundary_classi_call(Y,Z,V,A,steps,tol,factor,false) : pcghal_boundary_call(Y,Z,V,A,steps,tol,factor,false,"grad");
#else
 auto out=logistic ? pcghal_classi_call(Y,Z,V,A,steps,tol,factor,false) : pcghal_call(Y,Z,V,A,steps,tol,factor,false,"grad");
#endif
 for(int i=0;i<out.alphaiters.rows();i++) for(int j=0;j<k;j++) history[i*k+j]=out.alphaiters(i,j);
 *risk=out.risk;
#ifdef REPORT_STATUS
 std::strncpy(status,out.status.c_str(),127);
#else
 std::strncpy(status,"baseline",127);
#endif
 return out.alphaiters.rows();
 } catch(const std::exception& e){std::strncpy(status,e.what(),127); return -1;}
}
