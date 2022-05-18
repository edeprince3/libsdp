#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<string>

// rrsdp and bpsdp headers
#include<rrsdp_solver.h>
#include<bpsdp_solver.h>

// callback function to evaluate A.u
static void evaluate_Au(double * Au, double * u, void * data) {

    int dim = static_cast<int>(reinterpret_cast<intptr_t>(data));

    // evaluate left-hand side of A.x = b

    int off = 0;

    // Tr(D) = 1
    double dum = 0;
    for (size_t i = 0; i < dim; i++) {
        dum += u[i*dim + i];
    }
    Au[off++] = dum;

    // Dij + Qij = dij
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            Au[off++] = u[i*dim + j] + u[dim*dim + i*dim + j];
        }
    }

}

// callback function to evaluate A^T.u
static void evaluate_ATu(double * ATu, double * u, void * data) {

    int off = 0;

    int dim = static_cast<int>(reinterpret_cast<intptr_t>(data));

    memset((void*)ATu,'\0',2*dim*dim*sizeof(double));

    // Tr(D) = 1
    double dum = 0;
    for (size_t i = 0; i < dim; i++) {
        ATu[i*dim + i] += u[off];
    }
    off++;

    // Dij + Qij = dij
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            ATu[i*dim + j]           += u[off];
            ATu[dim*dim + i*dim + j] += u[off++];
        }
    }

}

// monitor function for bpsdp
static void bpsdp_monitor(int oiter, int iiter, double energy_primal, double energy_dual, double mu, double primal_error, double dual_error, void * data) {

    printf("      %5i %5i %11.6lf %11.6lf %11.6le %7.3lf %10.5le %10.5le\n",
        oiter,iiter,energy_primal,energy_dual,fabs(energy_primal-energy_dual),mu,primal_error,dual_error);

}

// monitor function for rrsdp
static void rrsdp_monitor(int oiter, int iiter, double lagrangian, double objective, double mu, double error, double zero, void * data) {

    printf("    %12i %12i %12.6lf %12.6lf %12.2le %12.3le\n",
                oiter,iiter,lagrangian,objective,mu,error);

}

void print_header(std::string sdp_solver_type) {

    printf("\n");

    if ( sdp_solver_type == "bpsdp" ) {

        printf("      oiter");
        printf(" iiter");
        printf("        E(p)");
        printf("        E(d)");
        printf("       E(gap)");
        printf("      mu");
        printf("      eps(p)");
        printf("      eps(d)\n");

    }else if ( sdp_solver_type == "rrsdp" ) {

        printf("           oiter");
        printf("        iiter");
        printf("            L");
        printf("            E");
        printf("           mu");
        printf("     ||Ax-b||\n");

    }

}

int main(int argc, char * argv[]) {

    if ( argc != 2 ) {
        printf("\n");
        printf("    usage: ./a.out sdpsolver (rrsdp / bpsdp)\n");
        printf("\n");
        exit(1);
    }

    std::string sdp_solver_type = argv[1];

    // 
    // consider the problem
    //
    // min Dij hij
    // 
    // with respect to Dij and subject to
    // 
    // Tr(D) = 1
    // Dij + Qij = dij
    // D >= 0
    // Q >= 0
    // 
    // in SDP language, we have
    // 
    // min x.c
    // 
    // subject to 
    // 
    // Ax = b
    // x >= 0
    // 
    // and we map D,Q -> x
    // and h -> c
    // and Tr(D) = 1, Dij + Qij = dij -> Ax  = b
    // 
    // for this example, lets say dim(D) = 100
    // 
    // so, n_primal would be 2 * 100 * 100 = 20000
    // and n_dual would be 1 + 100 * 100 = 10001
    // 
    // note that, for some reason, rrsdp and bpsdp find slightly different solutions that 
    // both appear to satisfy the simple constraints above
    // 

    size_t dim = 100;

    // dimension of primal solution vector (number of variables)
    size_t n_primal = 2 * dim * dim;

    // dimension of dual solution vector (number of constraints)
    size_t n_dual   = 1 + dim * dim;

    // the sdp solver
    std::shared_ptr<libsdp::SDPSolver> sdp;

    // need to define progress monitor function
    libsdp::SDPProgressMonitorFunction sdp_monitor;

    if ( sdp_solver_type == "bpsdp" ) {

        libsdp::SDPOptions sdp_options;
        sdp_options.sdp_objective_convergence = 1e-4;
        sdp_options.sdp_error_convergence     = 1e-4;
        sdp_options.cg_convergence            = 1e-8; 
        sdp_options.cg_maxiter                = 10000;
        sdp_options.maxiter                   = 10000;

        sdp = (std::shared_ptr<libsdp::SDPSolver>)(new libsdp::BPSDPSolver(n_primal,n_dual,sdp_options));

        sdp_monitor = bpsdp_monitor;

    }else if ( sdp_solver_type == "rrsdp" ) {

        libsdp::SDPOptions sdp_options;
        sdp_options.sdp_objective_convergence = 1e-4;
        sdp_options.sdp_error_convergence     = 1e-4;
        sdp_options.maxiter                   = 10000;
        //sdp_options.penalty_parameter_scaling = 0.99;

        sdp = (std::shared_ptr<libsdp::SDPSolver>)(new libsdp::RRSDPSolver(n_primal,n_dual,sdp_options));

        sdp_monitor = rrsdp_monitor;

    }else {

       printf("\n");
       printf("    invalid SDP solver\n");
       printf("\n");
       exit(1);

    }

    // container for primal solution vector. no need to initialize
    double * x = (double*)malloc(n_primal*sizeof(double));
    memset((void*)x,'\0',n_primal*sizeof(double));
    for (size_t i = 0; i < n_primal; i++) {
        x[i] = 2.0 * ( (double)rand()/RAND_MAX - 1.0 );
    }

    // container for vector defining problem
    double * c = (double*)malloc(n_primal*sizeof(double));
    memset((void*)c,'\0',n_primal*sizeof(double));
    
    // let's just consider hij with nearest-neighbor hopping terms
    for (size_t i = 0; i < dim - 1; i++) {
        c[i * dim + (i+1)] = 10.0;
    }
    for (size_t i = 1; i < dim; i++) {
        c[i * dim + (i-1)] = 10.0;
    }

    // container for constraints
    double * b = (double*)malloc(n_dual*sizeof(double));
    memset((void*)b,'\0',n_dual*sizeof(double));

    // dimensions of each block of x ... in this case, we have only 2 blocks, each with dimension 100
    std::vector<int> dimensions;
    dimensions.push_back(dim);
    dimensions.push_back(dim);

    // Tr(D) = 1
    b[0] = 1.0;

    // Dij + Qij = dij
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            size_t id = i * dim + j;
            b[1 + id] = (double)(i==j);
        }
    }

    // normally, you would use "data" to pass the details of your problem to the Au/ATu functions
    // for this problem, all we need is the dimension of the matrix D (dim)
    void *data = (void*)dim;

    // local maxiter allows you to kick out before bpsdp is done (this is sometimes useful)
    // this input is not used by rrsdp
    int local_maxiter = 10000;

    print_header(sdp_solver_type);

    sdp->solve(x,
               b,
               c,
               dimensions,
               local_maxiter,
               evaluate_Au,
               evaluate_ATu,
               sdp_monitor,
               data);

    printf("\n");

    for (size_t i = 0; i < n_primal; i++) {
        printf("%20.12lf\n",x[i]);
    }

}
