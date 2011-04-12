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

#ifndef MESH_H
#define MESH_H

#include "confdefs.h"

#include <deque>
#include <vector>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "Metis.h"
#include "Edge.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t, typename index_t> class Mesh{
 public:

  /*! 2D triangular mesh constructor. This is for use when there is no MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y){
    _init(NNodes, NElements, ENList, x, y, NULL, NULL, NULL);
  }
  
#ifdef HAVE_MPI  
  /*! 2D triangular mesh constructor. This is used when parallelised with MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param lnn2gnn mapping of local node numbering to global node numbering.
   * @param owner_range range of node id's owned by each partition.
   * @param mpi_comm the mpi communicator.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const index_t *lnn2gnn,
       const index_t *owner_range, MPI_Comm mpi_comm){
    _mpi_comm = mpi_comm;
    _init(NNodes, NElements, ENList, x, y, NULL, lnn2gnn, owner_range);
  }
#endif

  /*! 3D tetrahedra mesh constructor. This is for use when there is no MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const real_t *z){
    _init(NNodes, NElements, ENList, x, y, z, NULL, NULL);
  }

#ifdef HAVE_MPI
  /*! 3D tetrahedra mesh constructor. This is used when parallelised with MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   * @param lnn2gnn mapping of local node numbering to global node numbering.
   * @param owner_range range of node id's owned by each partition.
   * @param mpi_comm the mpi communicator.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const real_t *z, const index_t *lnn2gnn,
       const index_t *owner_range, MPI_Comm mpi_comm){
    _mpi_comm = mpi_comm;
    _init(NNodes, NElements, ENList, x, y, z, lnn2gnn, owner_range);
  }
#endif

  /*! Defragment mesh. This compresses the storage of internal data
    structures. This is useful if the mesh has been significently
    coarsened. */
  void defragment(std::map<index_t, index_t> *active_vertex_map=NULL){
    // Discover which verticies and elements are active.
    bool local_active_vertex_map=(active_vertex_map==NULL);
    if(local_active_vertex_map){
      active_vertex_map = new std::map<index_t, index_t>;
    }

    std::deque<index_t> active_vertex, active_element;
    for(size_t e=0;e<_NElements;e++){
      index_t nid = _ENList[e*nloc];
      if(nid<0)
        continue;
      active_element.push_back(e);

      (*active_vertex_map)[nid] = 0;
      for(size_t j=1;j<nloc;j++){
        nid = _ENList[e*nloc+j];
        (*active_vertex_map)[nid]=0;
      }
    }
    index_t cnt=0;
    for(typename std::map<index_t, index_t>::iterator it=active_vertex_map->begin();it!=active_vertex_map->end();++it){
      it->second = cnt++;
      active_vertex.push_back(it->first);
    }

    // Compress data structures.
    _NNodes = active_vertex.size();
    node_towner.resize(_NNodes);
    _NElements = active_element.size();
    element_towner.resize(_NElements);
    std::vector<index_t> defrag_ENList(_NElements*nloc);
    std::vector<real_t> defrag_coords(_NNodes*ndims);
    std::vector<real_t> defrag_metric(_NNodes*ndims*ndims);

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)_NElements;i++){
        index_t eid = active_element[i];
        for(size_t j=0;j<nloc;j++){
          defrag_ENList[i*nloc+j] = (*active_vertex_map)[_ENList[eid*nloc+j]];
        }
#ifdef _OPENMP
        element_towner[i] = omp_get_thread_num();
#else
        element_towner[i] = 0;
#endif
      }

      node_towner.resize(_NNodes);
#pragma omp for schedule(static)
      for(int i=0;i<(int)_NNodes;i++){
        index_t nid=active_vertex[i];
        for(size_t j=0;j<ndims;j++)
          defrag_coords[i*ndims+j] = _coords[nid*ndims+j];
        for(size_t j=0;j<ndims*ndims;j++)
          defrag_metric[i*ndims*ndims+j] = metric[nid*ndims*ndims+j];
#ifdef _OPENMP
        node_towner[i] = omp_get_thread_num();
#else
        node_towner[i] = 0;
#endif 
      }
    }

    _ENList.swap(defrag_ENList);
    _coords.swap(defrag_coords);
    metric.swap(defrag_metric);
    
    create_adjancy();
    calc_edge_lengths();

    if(local_active_vertex_map)
      delete active_vertex_map;

    element_towner.clear();
    node_towner.clear();
  }

  /// Add a new vertex
  index_t append_vertex(const real_t *x, const real_t *m){
    for(size_t i=0;i<ndims;i++)
      _coords.push_back(x[i]);

    for(size_t i=0;i<ndims*ndims;i++)
      metric.push_back(m[i]);

    node_towner.push_back(0);
    
    _NNodes++;
    
    return _NNodes-1;
  }

  /// Add a new element
  index_t append_element(const int *n){
    for(size_t i=0;i<nloc;i++)
      _ENList.push_back(n[i]);

    element_towner.push_back(0);

    _NElements++;
    
    return _NElements-1;
  }

  /// Erase an element
  void erase_element(const index_t eid){
    for(size_t i=0;i<nloc;i++)
      _ENList[eid*nloc+i] = -1;
  }

  /// Return the number of nodes in the mesh.
  int get_number_nodes() const{
    assert(_NNodes == (_coords.size()/ndims));
    return _NNodes;
  }

  /// Return the number of elements in the mesh.
  int get_number_elements() const{
    return _NElements;
  }

  /// Return the number of spatial dimensions.
  int get_number_dimensions() const{
    return ndims;
  }

#ifdef HAVE_MPI
  /// Return the MPI communicator.
  MPI_Comm get_mpi_comm() const{
    return _mpi_comm;
  }
#endif

  /// Return the thread that ownes the node.
  int get_node_towner(int i) const{
    if(node_towner.size())
      return node_towner[i];
    return 0;
  }

  /// Return the thread that ownes the element.
  int get_element_towner(int i) const{
    if(element_towner.size())
      return element_towner[i];
    return 0;
  }

  /// Return a pointer to the element-node list.
  const int *get_element(size_t eid) const{
    return &(_ENList[eid*nloc]);
  }

  /// Return the node id's connected to the specified node_id
  std::set<index_t> get_node_patch(index_t nid) const{
    std::set<index_t> patch;
    for(typename std::deque<index_t>::const_iterator it=NNList[nid].begin();it!=NNList[nid].end();++it)
      patch.insert(patch.end(), *it);
    return patch;
  }

  /// Grow a node patch around node id's until it reaches a minimum size.
  std::set<index_t> get_node_patch(index_t nid, size_t min_patch_size){
    std::set<index_t> patch = get_node_patch(nid);
    
    if(patch.size()<min_patch_size){
      std::set<index_t> front = patch, new_front;
      for(;;){
        for(typename std::set<index_t>::const_iterator it=front.begin();it!=front.end();it++){
          for(typename std::deque<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();jt++){
            if(patch.find(*jt)==patch.end()){
              new_front.insert(*jt);
              patch.insert(*jt);
            }
          }
        }
        
        if(patch.size()>=std::min(min_patch_size, _NNodes))
          break;
        
        front.swap(new_front);
      }
    }
    
    return patch;
  }

  /// Return positions vector.
  const real_t *get_coords(size_t nid) const{
    return &(_coords[nid*ndims]);
  }

  /// Return metric at that vertex.
  const real_t *get_metric(size_t nid) const{
    assert(metric.size()>0);
    return &(metric[nid*ndims*ndims]);
  }

  /// Return new local node number given on original node number.
  int new2old(int nid){
    return nid_new2old[nid];
  }

  /// Returns true if the node is in the halo.
  bool is_halo_node(int nid){
    return halo.count(nid)>0;
  }

  /// Default destructor.
  ~Mesh(){
  }

  /// Calculates the edge lengths in metric space.
  void calc_edge_lengths(){
    assert(Edges.size());
    
    for(typename std::set< Edge<real_t, index_t> >::iterator it=Edges.begin();it!=Edges.end();){
      typename std::set< Edge<real_t, index_t> >::iterator current_edge = it++;
      
      Edge<real_t, index_t> edge = *current_edge;
      Edges.erase(current_edge);
      
      index_t nid0 = edge.edge.first;
      index_t nid1 = edge.edge.second;
      
      edge.length = calc_edge_length(nid0, nid1);
      Edges.insert(edge);
    }
  }

  /// Calculates the edge lengths in metric space.
  real_t calc_edge_length(index_t nid0, index_t nid1){
    real_t length=-1.0;
    if(ndims==2){
      real_t ml00 = (metric[nid0*ndims*ndims]+metric[nid1*ndims*ndims])*0.5;
      real_t ml01 = (metric[nid0*ndims*ndims+1]+metric[nid1*ndims*ndims+1])*0.5;
      real_t ml11 = (metric[nid0*ndims*ndims+3]+metric[nid1*ndims*ndims+3])*0.5;
      
      real_t x=_coords[nid1*ndims]-_coords[nid0*ndims];
      real_t y=_coords[nid1*ndims+1]-_coords[nid0*ndims+1];
      
      length = sqrt((ml01*x + ml11*y)*y + 
                    (ml00*x + ml01*y)*x);
    }else{
      real_t ml00 = (metric[nid0*ndims*ndims  ]+metric[nid1*ndims*ndims  ])*0.5;
      real_t ml01 = (metric[nid0*ndims*ndims+1]+metric[nid1*ndims*ndims+1])*0.5;
      real_t ml02 = (metric[nid0*ndims*ndims+2]+metric[nid1*ndims*ndims+2])*0.5;
      
      real_t ml11 = (metric[nid0*ndims*ndims+4]+metric[nid1*ndims*ndims+4])*0.5;
      real_t ml12 = (metric[nid0*ndims*ndims+5]+metric[nid1*ndims*ndims+5])*0.5;
      
      real_t ml22 = (metric[nid0*ndims*ndims+8]+metric[nid1*ndims*ndims+8])*0.5;
      
      real_t x=_coords[nid1*ndims]-_coords[nid0*ndims];
      real_t y=_coords[nid1*ndims+1]-_coords[nid0*ndims+1];
      real_t z=_coords[nid1*ndims+2]-_coords[nid0*ndims+2];
      
      length = sqrt((ml02*x + ml12*y + ml22*z)*z + 
                    (ml01*x + ml11*y + ml12*z)*y + 
                    (ml00*x + ml01*y + ml02*z)*x);
    }
    return length;
  }
  
  real_t maximal_edge_length(){
    calc_edge_lengths();
    
    real_t L_max=0;
    for(typename std::set< Edge<real_t, index_t> >::const_iterator it=Edges.begin();it!=Edges.end();++it){
      L_max = std::max(L_max, it->length);
    }
    
    return L_max;
  }

 private:
  template<typename _real_t, typename _index_t> friend class MetricField;
  template<typename _real_t, typename _index_t> friend class Smooth;
  template<typename _real_t, typename _index_t> friend class Coarsen;
  template<typename _real_t, typename _index_t> friend class Refine;
  template<typename _real_t, typename _index_t> friend class Surface;
  template<typename _real_t, typename _index_t> friend class VTKTools;

  void _init(int NNodes, int NElements, const index_t *globalENList,
             const real_t *x, const real_t *y, const real_t *z,
             const index_t *lnn2gnn, const index_t *owner_range){
    mpi_nparts = 1;
    int rank=0;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_size(_mpi_comm, &mpi_nparts);
      MPI_Comm_rank(_mpi_comm, &rank);
    }
#endif

    numa_nparts = 1;
#ifdef HAVE_LIBNUMA
    numa_nparts =numa_max_node()+1;
#endif
    
    int etype;
    if(z==NULL){
      nloc = 3;
      ndims = 2;
      etype = 1; // METIS: triangles
    }else{
      nloc = 4;
      ndims = 3;
      etype = 2; // METIS: tetrahedra
    }

    _NNodes = NNodes;
    _NElements = NElements;

    // From the globalENList, create the halo and a local ENList if mpi_nparts>1.
    const index_t *ENList;
    std::map<index_t, index_t> gnn2lnn;
    if(mpi_nparts==1){
      ENList = globalENList;
    }else{
      assert(lnn2gnn!=NULL);
      for(size_t i=0;i<_NNodes;i++){
        gnn2lnn[lnn2gnn[i]] = i;
      }

      std::vector< std::set<int> > recv_set(mpi_nparts);
      index_t *localENList = new index_t[_NElements*nloc];
      for(size_t i=0;i<_NElements*nloc;i++){
        index_t gnn = globalENList[i];
        for(int j=0;j<mpi_nparts;j++){
          if(gnn<owner_range[j+1]){
            if(j!=rank)
              recv_set[j].insert(gnn);
            break;
          }
        }
        localENList[i] = gnn2lnn[gnn];
      }
      std::vector<int> recv_size(mpi_nparts);
      recv.resize(mpi_nparts);
      for(int j=0;j<mpi_nparts;j++){
        for(typename std::set<int>::const_iterator it=recv_set[j].begin();it!=recv_set[j].end();++it){
          recv[j].push_back(*it);
        }
        recv_size[j] = recv[j].size();
      }
      std::vector<int> send_size(mpi_nparts);
      MPI_Alltoall(&(recv_size[0]), 1, MPI_INT,
                   &(send_size[0]), 1, MPI_INT, _mpi_comm);
      
      // Setup non-blocking receives
      send.resize(mpi_nparts);      
      std::vector<MPI_Request> request(mpi_nparts*2);
      for(int i=0;i<mpi_nparts;i++){
        if((i==rank)||(send_size[i]==0)){
          request[i] =  MPI_REQUEST_NULL;
        }else{
          send[i].resize(send_size[i]);
          MPI_Irecv(&(send[i][0]), send_size[i], MPI_INT, i, 0, _mpi_comm, &(request[i]));
        }
      }
      
      // Non-blocking sends.
      for(int i=0;i<mpi_nparts;i++){
        if((i==rank)||(recv_size[i]==0)){
          request[mpi_nparts+i] =  MPI_REQUEST_NULL;
        }else{
          MPI_Isend(&(recv[i][0]), recv_size[i], MPI_INT, i, 0, _mpi_comm, &(request[mpi_nparts+i]));
        }
      }
      
      std::vector<MPI_Status> status(mpi_nparts*2);
      MPI_Waitall(mpi_nparts, &(request[0]), &(status[0]));
      MPI_Waitall(mpi_nparts, &(request[mpi_nparts]), &(status[mpi_nparts]));

      for(int j=0;j<mpi_nparts;j++){
        for(int k=0;k<recv_size[j];k++)
          recv[j][k] = gnn2lnn[recv[j][k]];
        
        for(int k=0;k<send_size[j];k++)
          send[j][k] = gnn2lnn[send[j][k]];
      }

      ENList = localENList;
    }

    _ENList.resize(_NElements*nloc);
    _coords.resize(_NNodes*ndims);

    // Partition the nodes and elements so that the mesh can be
    // topologically mapped to the computer node topology. If we have
    // NUMA library dev support then we use the number of memory
    // nodes. Otherwise, play it save and use the number of threads.
    std::vector<int> eid_new2old;
    std::vector<idxtype> epart(NElements, 0), npart(NNodes, 0);
    if(numa_nparts>1){
      int numflag = 0;
      int edgecut;
      
      std::vector<idxtype> metis_ENList(_NElements*nloc);
      for(size_t i=0;i<NElements*nloc;i++)
        metis_ENList[i] = ENList[i];
      METIS_PartMeshNodal(&NElements, &NNodes, &(metis_ENList[0]), &etype, &numflag, &numa_nparts,
                          &edgecut, &(epart[0]), &(npart[0]));
      metis_ENList.clear();

      // Create sets of nodes and elements in each partition
      std::vector< std::deque<int> > nodes(numa_nparts), elements(numa_nparts);
      for(int i=0;i<NNodes;i++)
        nodes[npart[i]].push_back(i);
      for(int i=0;i<NElements;i++)
        elements[epart[i]].push_back(i);
      
      std::vector< std::set<int> > edomains(numa_nparts);
      for(size_t i=0; i<_NElements; i++){
        edomains[epart[i]].insert(i);
      }
      
      // Create element renumbering
      for(int i=0;i<numa_nparts;i++){
        for(std::set<int>::const_iterator it=edomains[i].begin();it!=edomains[i].end();++it){
          eid_new2old.push_back(*it);
        }
      }
      
      // Create seperate graphs for each partition.
      std::vector< std::map<index_t, std::set<index_t> > > pNNList(numa_nparts);
      for(size_t i=0; i<_NElements; i++){
        for(size_t j=0;j<nloc;j++){
          int jnid = ENList[i*nloc+j];
          int jpart = npart[jnid];
          for(size_t k=j+1;k<nloc;k++){
            int knid = ENList[i*nloc+k];
            int kpart = npart[knid];
            if(jpart!=kpart)
              continue;
            pNNList[jpart][jnid].insert(knid);
            pNNList[jpart][knid].insert(jnid);
          }
        }
      }
      
      // Renumber nodes within each partition.
      for(int p=0;p<numa_nparts;p++){
        // Create mapping from node numbering to local thread partition numbering, and it's inverse.
        std::map<index_t, index_t> nid2tnid;
        std::deque<index_t> tnid2nid(pNNList[p].size());
        index_t loc=0;
        for(typename std::map<index_t, std::set<index_t> >::const_iterator it=pNNList[p].begin();it!=pNNList[p].end();++it){
          tnid2nid[loc] = it->first;
          nid2tnid[it->first] = loc++;
        }
        
        std::vector< std::set<index_t> > pgraph(nid2tnid.size());
        for(typename std::map<index_t, std::set<index_t> >::const_iterator it=pNNList[p].begin();it!=pNNList[p].end();++it){
          for(typename std::set<index_t>::const_iterator jt=it->second.begin();jt!=it->second.end();++jt){
            pgraph[nid2tnid[it->first]].insert(nid2tnid[*jt]);
          }
        }
        
        std::vector<int> porder;
        Metis<index_t>::reorder(pgraph, porder);
        
        for(typename std::vector<index_t>::const_iterator it=porder.begin();it!=porder.end();++it){
          nid_new2old.push_back(tnid2nid[*it]);
        }
      }
    }else{
      std::vector< std::set<index_t> > lNNList(_NNodes);
      for(size_t i=0; i<_NElements; i++){
        for(size_t j=0;j<nloc;j++){
          index_t nid_j = ENList[i*nloc+j];
          for(size_t k=j+1;k<nloc;k++){
            index_t nid_k = ENList[i*nloc+k];
            lNNList[nid_j].insert(nid_k);
            lNNList[nid_k].insert(nid_j);
          }
        }
      }
      Metis<index_t>::reorder(lNNList, nid_new2old);
      
      eid_new2old.resize(_NElements);
      for(size_t e=0;e<_NElements;e++)
        eid_new2old[e] = e;
    }
    
    // Reverse mapping of renumbering.
    std::vector<index_t> nid_old2new(_NNodes);
    for(size_t i=0;i<_NNodes;i++){
      nid_old2new[nid_new2old[i]] = i;
    }
    
    // Enforce first-touch policy
    element_towner.resize(_NElements);
    node_towner.resize(_NNodes);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)_NElements;i++){
        for(size_t j=0;j<nloc;j++){
          _ENList[i*nloc+j] = nid_old2new[ENList[eid_new2old[i]*nloc+j]];
        }
        element_towner[i] = epart[eid_new2old[i]];
      }
      if(ndims==2){
#pragma omp for schedule(static)
        for(int i=0;i<(int)_NNodes;i++){
          _coords[i*ndims  ] = x[nid_new2old[i]];
          _coords[i*ndims+1] = y[nid_new2old[i]];
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0;i<(int)_NNodes;i++){
          _coords[i*ndims  ] = x[nid_new2old[i]];
          _coords[i*ndims+1] = y[nid_new2old[i]];
          _coords[i*ndims+2] = z[nid_new2old[i]];
        }
      }
#pragma omp for schedule(static)
      for(size_t i=0;i<_NNodes;i++)
        node_towner[i] = npart[nid_new2old[i]];
    }

    if(mpi_nparts>1){
      // Take into account renumbering for halo.
      for(int j=0;j<mpi_nparts;j++){
        for(size_t k=0;k<recv[j].size();k++){
          int nid = nid_old2new[recv[j][k]];
          recv[j][k] = nid;
          halo.insert(nid);
        }
        for(size_t k=0;k<send[j].size();k++){
          int nid = nid_old2new[send[j][k]];
          send[j][k] = nid;
          halo.insert(nid);
        }
      }
    }

    if(mpi_nparts>1){
      delete [] ENList;
    }

    create_adjancy();
  }

  void halo_update(real_t *vec, int block){
#ifdef HAVE_MPI
    if(mpi_nparts<2)
      return;
    
    int rank;
    MPI_Comm_rank(_mpi_comm, &rank);

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(mpi_nparts*2);
    
    // Setup non-blocking receives.
    std::vector< std::vector<real_t> > recv_buff(mpi_nparts);
    for(int i=0;i<mpi_nparts;i++){
      if((i==rank)||(recv[i].size()==0)){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv[i].size()*block);  
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_DOUBLE, i, 0, _mpi_comm, &(request[i]));
      }
    }
    
    // Non-blocking sends.
    std::vector< std::vector<real_t> > send_buff(mpi_nparts);
    for(int i=0;i<mpi_nparts;i++){
      if((i==rank)||(send[i].size()==0)){
        request[mpi_nparts+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].begin();it!=send[i].end();++it)
          for(int j=0;j<block;j++){
            send_buff[i].push_back(vec[(*it)*block+j]);
          }
        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_DOUBLE, i, 0, _mpi_comm, &(request[mpi_nparts+i]));
      }
    }
    
    std::vector<MPI_Status> status(mpi_nparts*2);
    MPI_Waitall(mpi_nparts, &(request[0]), &(status[0]));
    MPI_Waitall(mpi_nparts, &(request[mpi_nparts]), &(status[mpi_nparts]));
    
    for(int i=0;i<mpi_nparts;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].begin();it!=recv[i].end();++it, ++k)
        for(int j=0;j<block;j++)
          vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }
#endif
  }

  void halo_update(index_t *vec, int block){
#ifdef HAVE_MPI
    if(mpi_nparts<2)
      return;
    
    int rank;
    MPI_Comm_rank(_mpi_comm, &rank);

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(mpi_nparts*2);
    
    // Setup non-blocking receives.
    std::vector< std::vector<index_t> > recv_buff(mpi_nparts);
    for(int i=0;i<mpi_nparts;i++){
      if((i==rank)||(recv[i].size()==0)){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv[i].size()*block);  
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_INT, i, 0, _mpi_comm, &(request[i]));
      }
    }
    
    // Non-blocking sends.
    std::vector< std::vector<index_t> > send_buff(mpi_nparts);
    for(int i=0;i<mpi_nparts;i++){
      if((i==rank)||(send[i].size()==0)){
        request[mpi_nparts+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].begin();it!=send[i].end();++it)
          for(int j=0;j<block;j++){
            send_buff[i].push_back(vec[(*it)*block+j]);
          }
        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_INT, i, 0, _mpi_comm, &(request[mpi_nparts+i]));
      }
    }
    
    std::vector<MPI_Status> status(mpi_nparts*2);
    MPI_Waitall(mpi_nparts, &(request[0]), &(status[0]));
    MPI_Waitall(mpi_nparts, &(request[mpi_nparts]), &(status[mpi_nparts]));
    
    for(int i=0;i<mpi_nparts;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].begin();it!=recv[i].end();++it, ++k)
        for(int j=0;j<block;j++)
          vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }
#endif
  }

  /// Create required adjancy lists.
  void create_adjancy(){
    // Create new NNList, NEList and edges
    std::vector< std::set<index_t> > NNList_set(_NNodes);
    NEList.clear();
    NEList.resize(_NNodes);
    Edges.clear();
    for(size_t i=0; i<_NElements; i++){
      for(size_t j=0;j<nloc;j++){
        index_t nid_j = _ENList[i*nloc+j];
        if(nid_j<0)
          break;
        NEList[nid_j].insert(i);
        for(size_t k=j+1;k<nloc;k++){
          index_t nid_k = _ENList[i*nloc+k];
          NNList_set[nid_j].insert(nid_k);
          NNList_set[nid_k].insert(nid_j);
          
          Edge<real_t, index_t> edge(nid_j, nid_k);
          typename std::set< Edge<real_t, index_t> >::iterator edge_ptr = Edges.find(edge);
          if(edge_ptr!=Edges.end()){
            edge.adjacent_elements = edge_ptr->adjacent_elements;
            Edges.erase(edge_ptr);
          }
          edge.adjacent_elements.insert(i);
          Edges.insert(edge);
        }
      }
    }
    
    // Compress NNList
    NNList.clear();
    NNList.resize(_NNodes);
    for(size_t i=0;i<_NNodes;i++){
      for(typename std::set<index_t>::const_iterator it=NNList_set[i].begin();it!=NNList_set[i].end();++it){
        NNList[i].push_back(*it);
      }
    }
  }

  size_t _NNodes, _NElements, ndims, nloc;
  std::vector<index_t> _ENList;
  std::vector<real_t> _coords;
  
  std::vector<index_t> nid_new2old;
  std::vector<int> element_towner, node_towner;

  // Adjancy lists
  std::vector< std::set<index_t> > NEList;
  std::vector< std::deque<index_t> > NNList;
  std::set< Edge<real_t, index_t> > Edges;

  // Metric tensor field.
  std::vector<real_t> metric;

  // Parallel support.
  int mpi_nparts, numa_nparts;
  std::vector< std::vector<int> > send, recv;
  std::set<int> halo;

#ifdef HAVE_MPI
  MPI_Comm _mpi_comm;
#endif
};
#endif
