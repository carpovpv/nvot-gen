
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/data.h>
#include <openbabel/obiter.h>
#include <openbabel/atom.h>
#include <openbabel/elements.h>

#include <random>
#include <thread>
#include <condition_variable>
#include <queue>
#include <algorithm>

//tcp/ip 
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <signal.h>
#include <fcntl.h>

const std::string fdir = "/home/pavel/dud/";

volatile bool HALT = false;
const int MapsCapacity = 100;
const int MaxProteinAtoms = 16384;
const int MaxLigandAtoms = 256;

const char *pred_protein;
bool mode_pred = false;

std::mutex maps_mutex, valid_mutex, ob_mutex;
std::condition_variable maps_empty, maps_full, val_empty, val_full;

struct MapItem 
{
   float y;
   float *x;
};

std::queue < struct MapItem > maps, val_maps;

std::mutex index_mutex;
int global_index = 0;
int valid_index = 0;
std::vector < int >inds;
std::vector < int >val_inds;

using namespace OpenBabel;
OBConversion conv;

enum AtomTypes {
    Hydrophobic = 0,
    Aromatic, HBA, HBD, PositiveI,
    NegativeI, Metal
};

const int Nd = 7 + 6;
const int cutoff = 8;
const double resolution = 0.5;
const int NN = 2 * cutoff / resolution;

int train_size;
int valid_size;
int pred_size;

struct ProteinComplex {
    std::string ligand;		//file name with the ligand
    int protein;		//index of the protein in the array proteins
    bool active;		//true - active ligand, false - decoy   
    int mode;                   //0 for training, 1 for validation and 2 for prediction
};

struct AtomData {
    OBMol mol;
    int num;			//number of atoms in the molecule
    float *x;
    float *y;
    float *z;
    char *n;			//atomic num
    char *is_aromatic;
    char *is_hbond_acceptor;
    char *is_hbond_donor;
    float *charge;
    float *wdv;
};

//generate rotation matrix
void createQ(const float *q, float *Q)
{
    float a = q[0];
    float b = q[1];
    float c = q[2];

    Q[0] = cos(a) * cos(b) * cos(c) - sin(a) * sin(b);
    Q[1] = -cos(a) * cos(b) * sin(c) - sin(a) * cos(c);
    Q[2] = cos(a) * sin(b);
   
    Q[3] = sin(a) * cos(b) * cos(c) + cos(a) * sin(c);
    Q[4] = - sin(a) * cos(b) * sin(c) + cos(a) * cos(c);
    Q[5] = sin(a) * sin(b);

    Q[6] = -sin(b) * cos(c);
    Q[7] = sin(b) * sin(c);
    Q[8] = cos(c);
}

__global__ void calcGrid(int off, float *x, float *y, float *z,
			 float *charge, float *vdw,
			 char *n, char *is_aromatic,
			 char *acceptor, char *donor,
			 int num, float x1, float y1, float z1,
			 float *grid)
{
    //indexes of our voxel
    int i_ = blockIdx.y;
    int j_ = blockIdx.x;
    int k_ = threadIdx.x;

    int threadId = ((i_ * NN + j_) * NN + k_) * Nd;

    float cx = x1 + i_ * resolution;
    float cy = y1 + j_ * resolution;
    float cz = z1 + k_ * resolution;

    //loop over all atoms 
    for (int i = 0; i < num; i++) {
	if (n[i] == 1)
	    continue;

	//grid[threadId + 0..Nd] our maps 
	float att[Nd];
	for (int o = 0; o < Nd; o++)
	    att[o] = 0.0;

	if (n[i] == 6) {
	    if (is_aromatic[i])
		att[Aromatic + off] = 1;
	    att[Hydrophobic + off] = 1;
	}

	if (acceptor[i])
	    att[HBA + off] = 1;
	if (donor[i])
	    att[HBD + off] = 1;
	if (charge[i] > 0)
	    att[PositiveI + off] = 1;
	else if (charge[i] < 0)
	    att[NegativeI + off] = 1;

	float value = 0.0;
	float dist =
	    sqrt((x[i] - cx) * (x[i] - cx) + (y[i] - cy) * (y[i] - cy) +
		 (z[i] - cz) * (z[i] - cz));

	if (dist < vdw[i] * 1.5) {
	    float h = 0.5 * vdw[i];
	    if (dist <= vdw[i]) {
		float ex = -dist * dist / (2 * h * h);
		value = exp(ex);
	    } else {
		float eval = 1.0 / (M_E * M_E);
		value =
		    dist * dist / (h * h) - 6.0 * eval * dist / h +
		    9.0 * eval;
	    }
	}

	float coulomb = charge[i] / (dist + 0.0001);

	for (int channel = off; channel < (off == 0 ? 7 : Nd); channel++) {
	    float v = value;
	    if (channel - off == PositiveI) {
		if (charge[i] > 0)
		    v = coulomb;
		else
		    v = 0.0;
	    }

	    if (channel - off == NegativeI) {
		if (charge[i] < 0)
		    v = -coulomb;
		else
		    v = 0.0;
	    }

	    grid[threadId + channel] += v * att[channel];
	}

    }

}

std::vector < AtomData > proteins;
std::vector < struct ProteinComplex >complexes;

void loadLigands(int cur_protein,
		 const std::string & prot_name, const std::string & folder, int pred)
{
    bool active = folder == "ligands";
    std::string ligdir = fdir + prot_name + "/" + folder + "/";

    DIR *ligands = opendir(ligdir.c_str());
    struct dirent *dl;

    while ((dl = readdir(ligands)) != NULL) {
	const char *file_name = dl->d_name;
	if (!strcmp(file_name, "..") || !strcmp(file_name, "."))
	    continue;
	if (strstr(file_name, "_out.pdbqt") == NULL)
	    continue;

	std::string fligand = ligdir + file_name;

	for(int lig =0; lig < (mode_pred ? 10 : (active ? 30 : 1)); lig++)
        {
    	    struct ProteinComplex item;
	    item.protein = cur_protein;
	    item.ligand = fligand;
	    item.active = active;
	    item.mode = pred;

    	    complexes.push_back(item);
	}
    }
    closedir(ligands);
}

void generate(bool training, int dev)
{
    cudaError_t err = cudaSetDevice(dev);
    printf("Error setting device: %d %s\n", err, cudaGetErrorName(err));

    const size_t grid_size = NN * NN * NN * Nd;

    //allocate device memory
    float *d_grid;
    float *d_x;
    float *d_y;
    float *d_z;
    float *d_charge;
    float *d_wdv;
    char *d_n;
    char *d_is_aromatic;
    char *d_hbond_acceptor;
    char *d_hbond_donor;

    cudaMalloc((void **) &d_grid, grid_size * sizeof(float));
    const int arr_size = MaxProteinAtoms * 3;

    cudaMalloc((void **) &d_x, arr_size * sizeof(float));
    cudaMalloc((void **) &d_y, arr_size * sizeof(float));
    cudaMalloc((void **) &d_z, arr_size * sizeof(float));
    cudaMalloc((void **) &d_charge, arr_size * sizeof(float));
    cudaMalloc((void **) &d_wdv, arr_size * sizeof(float));
    cudaMalloc((void **) &d_n, arr_size * sizeof(char));
    cudaMalloc((void **) &d_is_aromatic, arr_size * sizeof(char));
    cudaMalloc((void **) &d_hbond_acceptor, arr_size * sizeof(char));
    cudaMalloc((void **) &d_hbond_donor, arr_size * sizeof(char));

    //host memory
    float lig_x[MaxLigandAtoms];
    float lig_y[MaxLigandAtoms];
    float lig_z[MaxLigandAtoms];

    float o_x[MaxLigandAtoms];
    float o_y[MaxLigandAtoms];
    float o_z[MaxLigandAtoms];

    float lig_charge[MaxLigandAtoms];
    float lig_wdv[MaxLigandAtoms];
    char lig_n[MaxLigandAtoms];
    char lig_is_aromatic[MaxLigandAtoms];
    char lig_hbond_acceptor[MaxLigandAtoms];
    char lig_hbond_donor[MaxLigandAtoms];

    //calculating maps 

    while (!HALT) {

	int ind = -1;

	index_mutex.lock();
	if(training)
	{
	    global_index++;
	    if (global_index >= inds.size())
	       global_index = 0;
   	    ind = inds[global_index];
	}
	else
	{
	    valid_index++;
            if(valid_index >= val_inds.size())
		  valid_index = 0;
	    ind = val_inds[valid_index];
	}
	index_mutex.unlock();

        ob_mutex.lock();

	OBMol ligand;
	std::ifstream fn;

	fn.open(complexes[ind].ligand.c_str(), std::fstream::in);

	printf("Ligand: %s %d\n", complexes[ind].ligand.c_str(), ind);
	conv.SetInStream(&fn);
	conv.Read(&ligand);

	float l_x = 0.0;
	float l_y = 0.0;
	float l_z = 0.0;

	int cnt = 0;

	FOR_ATOMS_OF_MOL(a, ligand) {
	    l_x += a->GetX();
	    l_y += a->GetY();
	    l_z += a->GetZ();

	    cnt++;
	}

	ob_mutex.unlock();

	l_x /= cnt;
	l_y /= cnt;
	l_z /= cnt;

	double c_x = l_x;
	double c_y = l_y;
	double c_z = l_z;

        int n_augm = 1; //complexes[ind].active ? 10 : 1;

	for (int augm = 0; augm < n_augm; augm++) {

	    //generate random quaternion
	    float q[6];

	    const float max_angel = 0.2; //0.087626;  

	    //rotation
	    q[0] = (float) std::rand() / RAND_MAX * max_angel;
	    q[1] = (float) std::rand() / RAND_MAX * max_angel;
	    q[2] = (float) std::rand() / RAND_MAX * max_angel;

	    //translation
	    q[3] = (float) std::rand() / RAND_MAX - 0.5;
	    q[4] = (float) std::rand() / RAND_MAX - 0.5;
	    q[5] = (float) std::rand() / RAND_MAX - 0.5;

	    //generate quaternion 
	    float qv[9];
	    createQ(q, qv);

	    l_x = c_x;
	    l_y = c_y;
	    l_z = c_z;

	    //correct the center of the ligand 
	    l_x += q[3];
	    l_y += q[4];
	    l_z += q[5];

            printf("Ligand center: %g %g %g\n", l_x, l_y, l_z);

	    float x1, y1, z1;

	    //find the left corner of the cube
	    x1 = floor(l_x - cutoff);
	    y1 = floor(l_y - cutoff);
	    z1 = floor(l_z - cutoff);

	    int protId = complexes[ind].protein;
	    int arr_size = proteins[protId].num;

	    cudaMemset(d_grid, 0, grid_size * sizeof(float));

	    cudaMemcpy(d_x, proteins[protId].x, arr_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_y, proteins[protId].y, arr_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_z, proteins[protId].z, arr_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_n, proteins[protId].n, arr_size * sizeof(char),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_charge, proteins[protId].charge,
		       arr_size * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_wdv, proteins[protId].wdv,
		       arr_size * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_is_aromatic, proteins[protId].is_aromatic,
		       arr_size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_hbond_acceptor,
		       proteins[protId].is_hbond_acceptor,
		       arr_size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_hbond_donor, proteins[protId].is_hbond_donor,
		       arr_size * sizeof(char), cudaMemcpyHostToDevice);

	    //launch kernel on protein data 
	    dim3 _grid(NN, NN);
	    calcGrid <<< _grid, NN >>> (0, d_x, d_y, d_z, d_charge, d_wdv,
					d_n, d_is_aromatic,
					d_hbond_acceptor, d_hbond_donor,
					arr_size, x1, y1, z1, d_grid);

	    //ligand part 
	    cnt = 0;
	    ob_mutex.lock();
	    FOR_ATOMS_OF_MOL(a, ligand) {
		o_x[cnt] = a->GetX();
		o_y[cnt] = a->GetY();
		o_z[cnt] = a->GetZ();

		lig_n[cnt] = a->GetAtomicNum();
		lig_charge[cnt] = a->GetPartialCharge();
		lig_wdv[cnt] = OBElements::GetVdwRad(lig_n[cnt]);
		lig_is_aromatic[cnt] = a->IsAromatic()? 1 : 0;
		lig_hbond_acceptor[cnt] = a->IsHbondAcceptor()? 1 : 0;
		lig_hbond_donor[cnt] = a->IsHbondDonor()? 1 : 0;

		cnt++;
	    }

	    ob_mutex.unlock();

	    int lig_size = ligand.NumAtoms();

	    //set new coordinates
	    for (int i = 0; i < lig_size; ++i) {
		lig_x[i] = o_x[i];
		lig_y[i] = o_y[i];
		lig_z[i] = o_z[i];

		float x = o_x[i] - c_x;
		float y = o_y[i] - c_y;
		float z = o_z[i] - c_z;

		lig_x[i] =
		    (float) (qv[0] * x + qv[1] * y + qv[2] * z + q[3] +
			     c_x);
		lig_y[i] =
		    (float) (qv[3] * x + qv[4] * y + qv[5] * z + q[4] +
			     c_y);
		lig_z[i] =
		    (float) (qv[6] * x + qv[7] * y + qv[8] * z + q[5] +
			     c_z);
	    }

	    //copy to the GPU and calculate grids 
	    cudaMemcpy(d_x, lig_x, lig_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_y, lig_y, lig_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_z, lig_z, lig_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_n, lig_n, lig_size * sizeof(char),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_charge, lig_charge, lig_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_wdv, lig_wdv, lig_size * sizeof(float),
		       cudaMemcpyHostToDevice);
	    cudaMemcpy(d_is_aromatic, lig_is_aromatic,
		       lig_size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_hbond_acceptor, lig_hbond_acceptor,
		       lig_size * sizeof(char), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_hbond_donor, lig_hbond_donor,
		       lig_size * sizeof(char), cudaMemcpyHostToDevice);

	    //launch kernel on ligand data 
	    calcGrid <<< _grid, NN >>> (7, d_x, d_y, d_z, d_charge, d_wdv,
					d_n, d_is_aromatic,
					d_hbond_acceptor, d_hbond_donor,
					lig_size, x1, y1, z1, d_grid);

	    //label the data 
	    struct MapItem item;
            item.y = complexes[ind].active;
            item.x = (float *) malloc(grid_size * sizeof(float));

	    //copy the results back 
	    cudaMemcpy(item.x, d_grid, grid_size * sizeof(float),
		       cudaMemcpyDeviceToHost);


            if(training)
            {
		{
	        std::unique_lock < std::mutex > lk(maps_mutex);
    	        maps_empty.wait(lk,[] {
		   	    return maps.size() < MapsCapacity;
			    }
	        );
	        maps.push(item);
	        printf("New map was added, total number is %ld.\n",
		      maps.size());
                }
	        maps_full.notify_all();
	    }
            else 
	    {
		    {
                std::unique_lock < std::mutex > lv(valid_mutex);
    	        val_empty.wait(lv,[] {
		   	    return val_maps.size() < MapsCapacity;
			    }
	        );
	        val_maps.push(item);
	        printf("New map (valid)  was added, total number is %ld.\n",
		      val_maps.size());
                }
	        val_full.notify_all();
	    }

	    /*  
	       std::vector<FILE *> smap;

	       for(int i=0; i< Nd; i++)
	       {
	       char buf[64];
	       sprintf(buf, "test-%d.map", i);
	       printf("filename: %s\n", buf);

	       FILE * f = fopen(buf, "w");

	       fprintf(f, "GRID_PARAMETER_FILE\nGRID_DATA_FILE\nMACROMOLECULE\n");
	       fprintf(f, "SPACING 0.5\n");
	       fprintf(f, "NELEMENTS %d %d %d\n", NN-1, NN-1, NN-1);
	       fprintf(f, "CENTER %g %g %g\n", l_x, l_y, l_z);

	       smap.push_back(f);
	       }

	       for(int i=0; i< NN; i++)
	       for(int j=0; j< NN; j++)
	       for(int k=0; k< NN; k++)
	       for(int l = 0; l <Nd; l++)
	       fprintf(smap[l], "%g\n", grid[((i * NN + j) * NN + k) * Nd + l]);

	       for(int i =0; i<Nd; i++)
	       fclose(smap[i]);
	     */



	}			//augmentation cycle 

    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_n);
    cudaFree(d_charge);
    cudaFree(d_wdv);
    cudaFree(d_is_aromatic);
    cudaFree(d_hbond_acceptor);
    cudaFree(d_hbond_donor);

    cudaFree(d_grid);
}

int fd_is_valid(int fd)
{
    return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

void server(int port)
{

    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
	perror("socket failed");
	exit(EXIT_FAILURE);
    }
    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
		   &opt, sizeof(opt))) {
	perror("setsockopt");
	exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *) &address, sizeof(address)) < 0) {
	perror("bind failed");
	exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
	perror("listen");
	exit(EXIT_FAILURE);
    }

    const int batch = 10;
    const int num_elems = NN*NN*NN*Nd;

    float * data = (float *)malloc( sizeof(float) * (batch* (num_elems + 1) +1));
    if(data == NULL) 
       printf("Not enough memory\n");

    while (!HALT) {
	if ((new_socket = accept(server_fd, (struct sockaddr *) &address,
				 (socklen_t *) & addrlen)) < 0) {
	    perror("accept");
	    exit(EXIT_FAILURE);
	}

        while(true)
	{
           if(!fd_is_valid(new_socket))
	   {
		   printf("Error ins socket!\n");
		   break;
	   }


        char req[255];
	if( recv(new_socket, req, 255, 0) != 1) break;

        printf("Received command: %c\n", req[0]);

        auto cop = [&]()
	{
        char * c_data = (char *) data;

        float yy [batch];
        for(int mol = 0; mol < batch; mol++){
   
    	   struct MapItem it;
	   if(req[0] == 't') { it = maps.front();
	   maps.pop();}
	   else 
	   {
		it = val_maps.front();
		val_maps.pop();
	   }

           yy [mol] = it.y;
           memcpy(c_data + sizeof(float) * mol * num_elems, (const char *) it.x, sizeof(float) * num_elems);

	   free(it.x);
        }
              
        memcpy( c_data + sizeof(float) * batch * num_elems, (const char *) yy, sizeof(float) * batch);

        size_t sent = send(new_socket, (const char *) data, (batch * (num_elems +1) ) * sizeof(float), 0);
        printf("Sizes: %ld %ld\n", batch * sizeof(float), sent);    
 
	};

	if(req[0] == 't')
	{
		{
	        std::unique_lock<std::mutex> lk(maps_mutex);
		maps_full.wait(lk,[] {
			       return maps.size() >= 10;
			       }
		);
		cop();
                }
		maps_empty.notify_all();
	}
	else
	{
		{
    		std::unique_lock<std::mutex> lk(valid_mutex);
		val_full.wait(lk,[] {
			       return val_maps.size() >= 10;
			       }
		);
		cop();
                }
		val_empty.notify_all();
	}
	}
	close(new_socket);
    }

    close(server_fd);
    free(data);

}

int main(int argc, char **argv)
{
	/*
int nDevices = 0;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  return 0;
*/

    if(argc != 3) 
    {
	  printf("Usage: %s protein-for-prediction mode\n", argv[0]);
	  return EXIT_FAILURE;
    }	    

    pred_protein = argv[1];   
    mode_pred = (strcmp(argv[2], "prediction") == 0);

    signal(SIGPIPE, SIG_IGN);

    srand(time(NULL));
    conv.SetInFormat("PDBQT");

    DIR *prots = opendir(fdir.c_str());
    struct dirent *dp = NULL;

    while ((dp = readdir(prots)) != NULL) {
	const char *prot_name = dp->d_name;
	if (!strcmp(prot_name, "..") || !strcmp(prot_name, "."))
	    continue;

	int pred = 0;
	if (strcmp(pred_protein, prot_name) == 0)
	{
            if(mode_pred) pred = 0;		
	    else pred = 2;
	}
	else if(mode_pred) continue;

	std::string protein_name = fdir + prot_name + "/vina.pdbqt";

	struct AtomData item;
	std::ifstream fn;

	fn.open(protein_name.c_str(), std::fstream::in);
	conv.SetInStream(&fn);
	conv.Read(&item.mol);

	const int num = item.mol.NumAtoms();
	printf("Protein %s loaded with %d atoms.\n", prot_name, num);

	item.x = (float *) malloc(num * sizeof(float));
	item.y = (float *) malloc(num * sizeof(float));
	item.z = (float *) malloc(num * sizeof(float));
	item.n = (char *) malloc(num * sizeof(char));
	item.charge = (float *) malloc(num * sizeof(float));
	item.wdv = (float *) malloc(num * sizeof(float));
	item.is_aromatic = (char *) malloc(num * sizeof(char));
	item.is_hbond_acceptor = (char *) malloc(num * sizeof(char));
	item.is_hbond_donor = (char *) malloc(num * sizeof(char));
	item.num = num;

	int cur = 0;
	FOR_ATOMS_OF_MOL(a, item.mol) {
	    item.x[cur] = a->GetX();
	    item.y[cur] = a->GetY();
	    item.z[cur] = a->GetZ();
	    item.n[cur] = a->GetAtomicNum();
	    item.charge[cur] = a->GetPartialCharge();
	    item.is_aromatic[cur] = a->IsAromatic()? 1 : 0;
	    item.is_hbond_acceptor[cur] = a->IsHbondAcceptor()? 1 : 0;
	    item.is_hbond_donor[cur] = a->IsHbondDonor()? 1 : 0;
	    item.wdv[cur] = OBElements::GetVdwRad(item.n[cur]);

	    cur++;
	}

	int cur_protein = proteins.size();
	proteins.push_back(item);

	loadLigands(cur_protein, prot_name, "ligands", pred);
	loadLigands(cur_protein, prot_name, "decoys", pred);

	//break;
    }

    closedir(prots);
    
    const int total_size = complexes.size();
    int pred_size = 0;

    for(int i=0; i< total_size; i++)
	    if(complexes[i].mode == 2) pred_size++;
  
    valid_size = (total_size - pred_size) * (mode_pred ? 0 : 0.15);
    train_size = (total_size - pred_size - valid_size);

    printf("Total complexes: %d, training: %d, validation: %d, prediction: %d\n", total_size, train_size, valid_size, pred_size);

    //correct shuffle!

    global_index = -1;
    valid_index = -1;

    printf("Shuffling indexes.\n");
    for (int i = 0; i < total_size; i++) 
    {
	if(complexes[i].mode != 2) 
           inds.push_back(i);
    }

    if(!mode_pred)
    {
       std::random_device rd;
       std::mt19937 g(rd());
       std::shuffle(inds.begin(), inds.end(), g);

       for(int i =0; i< valid_size; i++)     
    	  val_inds.push_back(inds[i]);
    
       inds.erase(inds.begin(), inds.begin() + valid_size);
    }

    printf("Final indexes seize: %d %d\n", inds.size(), val_inds.size());

    if(mode_pred)
    {
       std::thread producer1(generate, true, 0);
       std::thread consumer1(server, 6222);

       producer1.join();
       consumer1.join();
    }
    else 
    {
       //spawn producers...
       std::thread producer1(generate, true, 0);
       std::thread producer2(generate, false, 0);

       std::thread producer3(generate, true, 1);
       std::thread producer4(generate, false, 1);
    
       //spawn consumer
       std::thread consumer1(server, 6222);
       std::thread consumer2(server, 6223);
   
       producer1.join();
       producer2.join();
       producer3.join();
       producer4.join();
     
       consumer1.join();
       consumer2.join();
    }

    printf("Normal exit.\n");

    //final cleanup
    for (int i = 0; i < proteins.size(); i++) {
	free(proteins[i].x);
	free(proteins[i].y);
	free(proteins[i].z);
	free(proteins[i].n);
	free(proteins[i].charge);
	free(proteins[i].is_aromatic);
	free(proteins[i].is_hbond_acceptor);
	free(proteins[i].is_hbond_donor);
	free(proteins[i].wdv);
    }

    return 0;
}
