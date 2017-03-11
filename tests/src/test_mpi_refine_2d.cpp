/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Gerard Gorman
 *  Applied Modelling and Computation Group
 *  Department of Earth Science and Engineering
 *  Imperial College London
 *
 *  g.gorman@imperial.ac.uk
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials provided
 *  with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 *  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#include <iostream>
#include <vector>
#include <unistd.h>

#include <errno.h>
#include <stdlib.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"

#include "Refine.h"
#include "ticker.h"

int main(int argc, char **argv)
{
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10.vtu");
    mesh->create_boundary();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    std::vector<double> psi(NNodes);
    double hmax=0.5,hmin=0.05;
    for(size_t i=0; i<NNodes; i++){
        double h = hmax*fabs(1-exp(-fabs(mesh->get_coords(i)[0]-0.5))) + hmin;
        double lbd = 1/(h*h);
        double lmax = 1/(hmax*hmax);
        psi[i] = lbd;//pow(mesh->get_coords(i)[0], 4) + pow(mesh->get_coords(i)[1], 4);
    }

    metric_field.add_field(&(psi[0]), 1/(hmax*hmax));//0.0001);
    metric_field.update_mesh();

    VTKTools<double>::export_vtu("../data/test_mpi_refine_2d_init", mesh);
        
    for (int iVer=0; iVer<mesh->get_number_nodes(); ++iVer){
      const double * coords = mesh->get_coords(iVer);
      printf("DEBUG(%d)  vertex[%d (%d)]  %1.2f %1.2f\n", rank, iVer, mesh->get_globalNodeNumbering(iVer), coords[0], coords[1]);
    }
    
    for (int iTri=0; iTri<mesh->get_number_elements(); ++iTri){
        const int * tri = mesh->get_element(iTri);
        printf("DEBUG(%d)  triangle[%d]  %d %d %d\n", rank, iTri, tri[0], tri[1], tri[2]);
    }
    
    for (int iVer=0; iVer<mesh->get_number_nodes(); ++iVer){
        const double * met = mesh->get_metric(iVer);
        printf("DEBUG(%d)  metric[%d]  %1.2f %1.2f %1.2f\n", rank, iVer, met[0], met[1], met[2]);
    }

    Refine<double,2> adapt(*mesh);

    double tic = get_wtime();
    for(int i=0; i<3; i++)
        adapt.refine(sqrt(2.0), 0);
    double toc = get_wtime();

    if(!mesh->verify()) {
        std::cout<<"ERROR(rank="<<rank<<"): Verification failed after refinement.\n";
    }

    mesh->defragment();

    VTKTools<double>::export_vtu("../data/test_mpi_refine_2d", mesh);

    delete mesh;

    if(rank==0) {
        std::cout<<"Refine time = "<<toc-tic<<std::endl;
        std::cout<<"pass"<<std::endl;
    }
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
