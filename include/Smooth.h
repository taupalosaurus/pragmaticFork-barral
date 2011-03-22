/*
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a full list
 *    of copyright holders.
 *
 *    Gerard Gorman
 *    Applied Modelling and Computation Group
 *    Department of Earth Science and Engineering
 *    Imperial College London
 *
 *    amcgsoftware@imperial.ac.uk
 *    
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation,
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
 *    USA
 */

#ifndef SMOOTH_H
#define SMOOTH_H

#include <omp.h>
#include <set>
#include <map>
#include <vector>
#include <deque>

#include "ElementProperty.h"
#include "Surface.h"
#include "Mesh.h"
#include "Colour.h"
#include "MetricTensor.h"

/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t, typename index_t>
  class Smooth{
 public:
  /// Default constructor.
  Smooth(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
    _NNodes = mesh.get_number_nodes();
    _NElements = mesh.get_number_elements();
    _ndims = mesh.get_number_dimensions();
    _nloc = (_ndims==2)?3:4;
    _ENList = mesh.get_enlist();;
    _coords = mesh.get_coords();
#ifdef HAVE_MPI
    msh_comm = mesh.get_mpi_comm();
#endif
    _surface = &surface;
    msh = &mesh;
    init_cache();
  }

  /// Default destructor.
  ~Smooth(){
  }
  
  void init_cache(){
    std::vector<index_t> colour(_NNodes, 0);
    if(omp_get_max_threads()>1)
      Colour<index_t>::greedy(msh->NNList, &(colour[0]));
    
    for(int i=0;i<_NNodes;i++)
      colour_sets[colour[i]].push_back(i);

    return;
  }

  real_t smooth(bool qconstrain=false){
    real_t qlinfinity = std::numeric_limits<real_t>::max();
    real_t qmean = 0.0, qrms=0.0;
    
    int ncolours = colour_sets.size();

    if(_ndims==2){
      // Smoothing loop.
      const real_t *refx0 = _coords + _ENList[0]*_ndims;
      const real_t *refx1 = _coords + _ENList[1]*_ndims;
      const real_t *refx2 = _coords + _ENList[2]*_ndims;
      ElementProperty<real_t> property(refx0, refx1, refx2);
      
      for(int colour=0; colour<ncolours; colour++){
#pragma omp parallel
        {
          int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[colour][cn];
            real_t min_q=0, mean_q=0;
            if(qconstrain){
              typename std::set<index_t>::iterator ie=msh->NEList[node].begin();
              {
                const index_t *n=_ENList+(*ie)*3;
                real_t *x0 = _coords + n[0]*_ndims;
                real_t *x1 = _coords + n[1]*_ndims;
                real_t *x2 = _coords + n[2]*_ndims;
                min_q = property.lipnikov(x0, x1, x2, 
                                          &(msh->metric[n[0]*4]),
                                          &(msh->metric[n[1]*4]),
                                          &(msh->metric[n[2]*4]));
                mean_q = min_q;
              }
              for(;ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*3;
                real_t *x0 = _coords + n[0]*_ndims;
                real_t *x1 = _coords + n[1]*_ndims;
                real_t *x2 = _coords + n[2]*_ndims;
                real_t q = property.lipnikov(x0, x1, x2,
                                             &(msh->metric[n[0]*4]),
                                             &(msh->metric[n[1]*4]),
                                             &(msh->metric[n[2]*4]));
                min_q = min(q, min_q);
                mean_q += q;
              }
              mean_q/=msh->NEList[node].size();
            }
            real_t A00=0, A01=0, A11=0, q0=0, q1=0;
            for(typename std::deque<index_t>::const_iterator il=msh->NNList[node].begin();il!=msh->NNList[node].end();++il){
              real_t ml00 = 0.5*(msh->metric[node*4  ] + msh->metric[*il*4  ]);
              real_t ml01 = 0.5*(msh->metric[node*4+1] + msh->metric[*il*4+1]);
              real_t ml11 = 0.5*(msh->metric[node*4+3] + msh->metric[*il*4+3]);
              
              q0 += ml00*get_x(*il) + ml01*get_y(*il);
              q1 += ml01*get_x(*il) + ml11*get_y(*il);
            
              A00 += ml00;
              A01 += ml01;
              A11 += ml11;
            }
            // Want to solve the system Ap=q to find the new position, p.
            real_t p[] = {-(pow(A01, 2)/((pow(A01, 2)/A00 - A11)*pow(A00, 2)) - 1/A00)*q0 + 
                          A01*q1/((pow(A01, 2)/A00 - A11)*A00),
                          A01*q0/((pow(A01, 2)/A00 - A11)*A00) - q1/(pow(A01, 2)/A00 - A11)};
          
            if(_surface->contains_node(node)){
              // If this node is on the surface then we have to project
              // this position back onto the surface.
              std::set<index_t> *patch;
              patch = new std::set<index_t>;
              *patch = _surface->get_surface_patch(node);
              
              std::set<int> *coids;
              coids = new std::set<int>;
              
              for(typename std::set<index_t>::const_iterator e=patch->begin();e!=patch->end();++e)
                coids->insert(_surface->get_coplanar_id(*e));
              
              if(coids->size()<2){
                const real_t *normal = _surface->get_normal(*patch->begin());
                p[0] -= (p[0]-get_x(node))*fabs(normal[0]);
                p[1] -= (p[1]-get_y(node))*fabs(normal[1]);
              }
              
              delete coids;
              delete patch;
              
              if(coids->size()>1){
                // Test if this is a corner node, in which case it cannot be moved.
                continue;
              }
            }
          
            // Interpolate metric at this new position.
            real_t mp[4], l[3];
            int best_e=*msh->NEList[node].begin();
            for(size_t b=0;b<5;b++){
              real_t tol=-1;
              for(typename std::set<index_t>::iterator ie=msh->NEList[node].begin();ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*3;
              
                real_t *x0 = _coords + n[0]*_ndims;
                real_t *x1 = _coords + n[1]*_ndims;
                real_t *x2 = _coords + n[2]*_ndims;
              
                l[0] = property.area(p,  x1, x2);
                l[1] = property.area(x0, p,  x2);
                l[2] = property.area(x0, x1, p);
              
                real_t min_l = min(l[0], min(l[1], l[2]));
                if(min_l>tol){
                  tol = min_l;
                  best_e = *ie;
                }
                if(tol>=0){
                  break;
                }
              }
              if(tol>=0){
                break;
              }else{
                p[0] = (get_x(node)+p[0])/2;
                p[1] = (get_y(node)+p[1])/2;
              }
            }
            if((l[0]<0)||(l[1]<0)||(l[2]<0))
              continue;
          
            {
              const index_t *n=_ENList+best_e*3;
              real_t *x0 = _coords + n[0]*_ndims;
              real_t *x1 = _coords + n[1]*_ndims;
              real_t *x2 = _coords + n[2]*_ndims;
              real_t L = property.area(x0, x1, x2);
              if(L<0){
                std::cerr<<"negative area :: "<<node<<", "<<L<<std::endl;
              }
              for(size_t i=0;i<4;i++)
                mp[i] = (l[0]*msh->metric[n[0]*4+i]+
                         l[1]*msh->metric[n[1]*4+i]+
                         l[2]*msh->metric[n[2]*4+i])/L;
              
              MetricTensor<real_t>::positive_definiteness(2, mp);
            }
            
            bool improvement = true;
            if(qconstrain){
              // Check if this positions improves the local mesh quality.
              real_t min_q_new = std::numeric_limits<real_t>::max(), mean_q_new=0;
              for(typename std::set<index_t>::iterator ie=msh->NEList[node].begin();ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*_nloc;
                int iloc = 0;
                while(n[iloc]!=(int)node){
                  iloc++;
                }
                int loc1 = (iloc+1)%3;
                int loc2 = (iloc+2)%3;
                
                const real_t *x1 = _coords + n[loc1]*_ndims;
                const real_t *x2 = _coords + n[loc2]*_ndims;
                
                real_t q = property.lipnikov(p, x1, x2, 
                                             mp,
                                             &(msh->metric[n[loc1]*4]),
                                             &(msh->metric[n[loc2]*4]));
                mean_q_new += q;
                min_q_new = min(min_q_new, q);
              }
              mean_q_new /= msh->NEList[node].size();
              
              improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
            }  
            if(improvement){
              for(int j=0;j<_ndims;j++)
                _coords[node*_ndims+j] = p[j];
                
              msh->metric[(node)*4  ] = mp[0];
              msh->metric[(node)*4+1] = mp[1];
              msh->metric[(node)*4+2] = mp[1];
              msh->metric[(node)*4+3] = mp[3];
            }
          }
        }
      }

      std::vector<real_t> qvec(_NElements);
#pragma omp parallel
      {
        real_t lqlinfinity = std::numeric_limits<real_t>::max();
#pragma omp for schedule(static) reduction(+:qmean)
        for(int i=0;i<_NElements;i++){
          const int *n=_ENList+i*3;
          const real_t *x0 = _coords + n[0]*_ndims;
          const real_t *x1 = _coords + n[1]*_ndims;
          const real_t *x2 = _coords + n[2]*_ndims;
          
          qvec[i] = property.lipnikov(x0, x1, x2,
                                      &(msh->metric[n[0]*4]),
                                      &(msh->metric[n[1]*4]),
                                      &(msh->metric[n[2]*4]));
          lqlinfinity = std::min(lqlinfinity, qvec[i]);
          qmean += qvec[i]/_NElements;
        }
#pragma omp for schedule(static) reduction(+:qrms)
        for(int i=0;i<_NElements;i++){
          qrms += pow(qvec[i]-qmean, 2);
        }
#pragma omp critical 
        {
          qlinfinity = std::min(qlinfinity, lqlinfinity);
        }
      }
    }else{
      // Smoothing loop.
      const real_t *refx0 = _coords + _ENList[0]*_ndims;
      const real_t *refx1 = _coords + _ENList[1]*_ndims;
      const real_t *refx2 = _coords + _ENList[2]*_ndims;
      const real_t *refx3 = _coords + _ENList[3]*_ndims;
      ElementProperty<real_t> property(refx0, refx1, refx2, refx3);
      
      for(int colour=0; colour<ncolours; colour++){
#pragma omp parallel
        {
          int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[colour][cn];
            real_t min_q=0, mean_q=0;
            if(qconstrain){
              typename std::set<index_t>::iterator ie=msh->NEList[node].begin();
              {
                const index_t *n=_ENList+(*ie)*4;
                real_t *x0 = _coords + n[0]*_ndims;
                real_t *x1 = _coords + n[1]*_ndims;
                real_t *x2 = _coords + n[2]*_ndims;
                real_t *x3 = _coords + n[3]*_ndims;
                min_q = property.lipnikov(x0, x1, x2, x3,
                                          &(msh->metric[n[0]*9]),
                                          &(msh->metric[n[1]*9]),
                                          &(msh->metric[n[2]*9]),
                                          &(msh->metric[n[3]*9]));
                mean_q = min_q;
              }
              for(;ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*4;
                real_t *x0 = _coords + n[0]*_ndims;
                real_t *x1 = _coords + n[1]*_ndims;
                real_t *x2 = _coords + n[2]*_ndims;
                real_t *x3 = _coords + n[3]*_ndims;
                real_t q = property.lipnikov(x0, x1, x2, x3,
                                             &(msh->metric[n[0]*9]),
                                             &(msh->metric[n[1]*9]),
                                             &(msh->metric[n[2]*9]),
                                             &(msh->metric[n[3]*9]));
                min_q = min(q, min_q);
                mean_q += q;
              }
              mean_q/=msh->NEList[node].size();
            }
            real_t A00=0, A01=0, A02=0, A11=0, A12=0, A22=0, q0=0, q1=0, q2=0;
            for(typename std::deque<index_t>::const_iterator il=msh->NNList[node].begin();il!=msh->NNList[node].end();++il){
              real_t ml00 = 0.5*(msh->metric[node*9  ] + msh->metric[*il*9  ]);
              real_t ml01 = 0.5*(msh->metric[node*9+1] + msh->metric[*il*9+1]);
              real_t ml02 = 0.5*(msh->metric[node*9+2] + msh->metric[*il*9+2]);
              real_t ml11 = 0.5*(msh->metric[node*9+4] + msh->metric[*il*9+4]);
              real_t ml12 = 0.5*(msh->metric[node*9+5] + msh->metric[*il*9+5]);
              real_t ml22 = 0.5*(msh->metric[node*9+8] + msh->metric[*il*9+8]);
            
              q0 += ml00*get_x(*il) + ml01*get_y(*il) + ml02*get_z(*il);
              q1 += ml01*get_x(*il) + ml11*get_y(*il) + ml12*get_z(*il);
              q2 += ml02*get_x(*il) + ml12*get_y(*il) + ml22*get_z(*il);
            
              A00 += ml00;
              A01 += ml01;
              A02 += ml02;
              A11 += ml11;
              A12 += ml12;
              A22 += ml22;
            }
            // Want to solve the system Ap=q to find the new position, p.
            real_t p[] = {-(((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*(A01*A02/A00 - A12)/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - A01/((A01*A01/A00 - A11)*A00))*q1 + (pow((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00, 2)/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22) - A01*A01/((A01*A01/A00 - A11)*A00*A00) + 1/A00)*q0 + ((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*q2/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22),
                          (pow(A01*A02/A00 - A12, 2)/(pow(A01*A01/A00 - A11, 2)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - 1/(A01*A01/A00 - A11))*q1 - (((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*(A01*A02/A00 - A12)/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - A01/((A01*A01/A00 - A11)*A00))*q0 - (A01*A02/A00 - A12)*q2/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)),
                          ((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*q0/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22) - (A01*A02/A00 - A12)*q1/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) + q2/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)};
          
            if(_surface->contains_node(node)){
              // If this node is on the surface then we have to project
              // this position back onto the surface.
              std::set<index_t> *patch;
              patch = new std::set<index_t>;
              *patch = _surface->get_surface_patch(node);

              std::map<int, std::set<int> > *coids;
              coids = new std::map<int, std::set<int> >;

              for(typename std::set<index_t>::const_iterator e=patch->begin();e!=patch->end();++e)
                (*coids)[_surface->get_coplanar_id(*e)].insert(*e);

              if(coids->size()<3)
                for(std::map<int, std::set<int> >::const_iterator ic=coids->begin();ic!=coids->end();++ic){
                  const real_t *normal = _surface->get_normal(*(ic->second.begin()));
                  p[0] -= (p[0]-get_x(node))*fabs(normal[0]);
                  p[1] -= (p[1]-get_y(node))*fabs(normal[1]);
                  p[2] -= (p[2]-get_z(node))*fabs(normal[2]);
                }
              
              size_t coids_size = coids->size();
              
              delete patch;
              delete coids;
              
              // Test if this is a corner node, or edge node in which case it cannot be moved.
              if(coids_size>2)
                continue;
            }

            // Interpolate metric at this new position.
            real_t mp[9], l[4];
            int best_e=*msh->NEList[node].begin();
            bool inverted=false;
            for(size_t bisections=0;bisections<5;bisections++){ // 5 bisections along the search line
              real_t tol=-1;
              for(typename std::set<index_t>::iterator ie=msh->NEList[node].begin();ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*4;
                real_t vectors[] = {get_x(n[0]), get_y(n[0]), get_z(n[0]),
                                    get_x(n[1]), get_y(n[1]), get_z(n[1]),
                                    get_x(n[2]), get_y(n[2]), get_z(n[2]),
                                    get_x(n[3]), get_y(n[3]), get_z(n[3])};
                real_t *x0 = vectors;
                real_t *x1 = vectors+3;
                real_t *x2 = vectors+6;
                real_t *x3 = vectors+9;
              
                real_t *r[4];
                for(int iloc=0;iloc<4;iloc++)
                  if(n[iloc]==(node)){
                    r[iloc] = p;
                  }else{
                    r[iloc] = vectors+3*iloc;
                  }
                real_t volume = property.volume(r[0], r[1], r[2], r[3]);
                if(volume<=0){
                  inverted = true;
                  break;
                }
              
                if(tol<0){
                  real_t L = property.volume(x0, x1, x2, x3);
                  if(L<0)
                    std::cerr<<"negative volume :: "<<node<<", "<<L<<std::endl;
                
                  l[0] = property.volume(p,  x1, x2, x3)/L;
                  l[1] = property.volume(x0, p,  x2, x3)/L;
                  l[2] = property.volume(x0, x1, p,  x3)/L;
                  l[3] = property.volume(x0, x1, x2, p)/L;
                
                  real_t min_l = min(min(l[0], l[1]), min(l[2], l[3]));
                  if(min_l>tol){
                    tol = min_l;
                    best_e = *ie;
                  }
                }
              }
              if(inverted){
                p[0] = (get_x(node)+p[0])/2;
                p[1] = (get_y(node)+p[1])/2;
                p[2] = (get_z(node)+p[2])/2;
              }else{
                break;
              }
            }
          
            if(inverted)
              continue;
          
            {
              const index_t *n=_ENList+best_e*4;
              for(size_t i=0;i<9;i++)
                mp[i] =
                  l[0]*msh->metric[n[0]*9+i]+
                  l[1]*msh->metric[n[1]*9+i]+
                  l[2]*msh->metric[n[2]*9+i]+
                  l[3]*msh->metric[n[3]*9+i];
              
              MetricTensor<real_t>::positive_definiteness(3, mp);
            }
          
            bool improvement=true;
            if(qconstrain){
              // Check if this positions improves the local mesh quality.
              real_t min_q_new = std::numeric_limits<real_t>::max(), mean_q_new=0;
              for(typename std::set<index_t>::iterator ie=msh->NEList[node].begin();ie!=msh->NEList[node].end();++ie){
                const index_t *n=_ENList+(*ie)*_nloc;
                real_t vectors[] = {get_x(n[0]), get_y(n[0]), get_z(n[0]),
                                    get_x(n[1]), get_y(n[1]), get_z(n[1]),
                                    get_x(n[2]), get_y(n[2]), get_z(n[2]),
                                    get_x(n[3]), get_y(n[3]), get_z(n[3])};
                
                real_t *r[4], *m[4];
                for(int iloc=0;iloc<4;iloc++)
                  if(n[iloc]==(node)){
                    r[iloc] = p;
                    m[iloc] = mp;
                  }else{
                    r[iloc] = vectors+3*iloc;
                    m[iloc] = &(msh->metric[n[iloc]*9]);
                  }
                real_t q = property.lipnikov(r[0], r[1], r[2], r[3],
                                             m[0], m[1], m[2], m[3]);
                mean_q_new += q;
                min_q_new = min(min_q_new, q);
              }
              
              mean_q_new /= msh->NEList[node].size();
              
              improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
            }
            if(improvement){
              for(int j=0;j<_ndims;j++)
                _coords[node*_ndims+j] = p[j];

              msh->metric[(node)*9  ] = mp[0]; msh->metric[(node)*9+1] = mp[1]; msh->metric[(node)*9+2] = mp[2];
              msh->metric[(node)*9+3] = mp[1]; msh->metric[(node)*9+4] = mp[4]; msh->metric[(node)*9+5] = mp[5];
              msh->metric[(node)*9+6] = mp[2]; msh->metric[(node)*9+7] = mp[5]; msh->metric[(node)*9+8] = mp[8];
            }
          }
        }
      }
      
      std::vector<real_t> qvec(_NElements);
#pragma omp parallel
      {
        real_t lqlinfinity = std::numeric_limits<real_t>::max();
#pragma omp for schedule(static) reduction(+:qmean)
        for(int i=0;i<_NElements;i++){
          const int *n=_ENList+i*4;
          const real_t *x0 = _coords + n[0]*_ndims;
          const real_t *x1 = _coords + n[1]*_ndims;
          const real_t *x2 = _coords + n[2]*_ndims;
          const real_t *x3 = _coords + n[3]*_ndims;
          
          qvec[i] = property.lipnikov(x0, x1, x2, x3,
                                      &(msh->metric[n[0]*9]),
                                      &(msh->metric[n[1]*9]),
                                      &(msh->metric[n[2]*9]),
                                      &(msh->metric[n[3]*9]));

          lqlinfinity = std::min(lqlinfinity, qvec[i]);
          qmean += qvec[i]/_NElements;
        }
#pragma omp for schedule(static) reduction(+:qrms)
        for(int i=0;i<_NElements;i++){

          qrms += pow(qvec[i]-qmean, 2);
        }
#pragma omp critical 
        {
          qlinfinity = std::min(qlinfinity, lqlinfinity);
        }
      }
    }
    
    qrms=sqrt(qrms/_NElements);

    // std::cout<<_NElements<<" "<<qmean<<" "<<qrms<<" "<<qlinfinity<<std::endl;
    return qmean;
  }

 private:
  inline real_t get_x(index_t nid){
    return _coords[nid*_ndims];
  }

  inline real_t get_y(index_t nid){
    return _coords[nid*_ndims+1];
  }

  inline real_t get_z(index_t nid){
    return _coords[nid*_ndims+2];
  }

  int _NNodes, _NElements, _ndims, _nloc;
  const index_t *_ENList, *_node_distribution;
  real_t *_coords;
  std::vector<int> norder;
  std::set<index_t> surface_nodes;
  std::map<int, std::deque<index_t> > colour_sets;

  Surface<real_t, index_t> *_surface;
  Mesh<real_t, index_t> *msh;
};
#endif