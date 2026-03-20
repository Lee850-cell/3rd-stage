#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    

    const int tag = 123;
    
    std::vector<float> a(15, 0.0f);

    MPI_Request request;
    MPI_Status status;


    if (rank == 0){

        for (int i = 0; i<10; ++i) a[i] = i * 1.0f;

            MPI_Isend(a.data(), 10, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &request);
            std::cout << "[Rank 0] Isend initiated.\n";

            MPI_Wait(&request, &status);
            std::cout <<"[Rank 0] Wait finished. Send is complete.\n";
        
    }else if (rank == 1){
        MPI_Irecv(a.data(),15, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &request);
        
        std::cout << "[Rank 1] Irecv initiated.\n";

        MPI_Wait(&request, &status);
        std::cout <<"[Rank 1] Wait finished. Data is ready to use.\n";
    }

        
    MPI_Finalize();
    return 0;

}




