// 
// pi -- for
//
#include <omp.h>
#include <stdio.h>
#include <math.h>

const int n=10000;
const int nthreads=4;
 
static inline double f(double x)
{
	return 4/(x*x+1);
}

int main(void)
{
	double a=0.0, b=1.0;
	double h=(b-a)/n;
	double mypi[nthreads];
	int i,tid;
	
	#pragma omp parallel num_threads(nthreads) private(i,tid) shared(a,h,mypi)
	{
		tid = omp_get_thread_num(); 
		
		#pragma omp for
		for(i=1; i<n; i++)
		{
			mypi[tid] += f(a+i*h);
		}
	}
	
	for(i=1; i<nthreads; i++)
		mypi[0] += mypi[i];
	
	mypi[0] += (f(a) + f(b))/2;
	mypi[0] = h*mypi[0];
	
	printf("nthreads=%d, mypi=%.10f, err=%.2e\n", 
		nthreads, mypi[0], fabs(mypi[0]-4*atan(1)));

	return 0;
}