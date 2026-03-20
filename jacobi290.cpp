#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 100;
    const int tag = 999;
    std::vector<double> buf1(count,0.0f);
    std::vector<double> buf2(count,0.0f);

    if (rank == 0){

        int bsize;
        MPI_Pack_size(count, MPI_FLOAT,MPI_COMM_WORLD, &bsize);
        int total_buffer_size = 2 * (bsize + MPI_BSEND_OVERHEAD);
        std::vector<char> paking_buffer(total_buffer_size);
        MPI_Buffer_attach(paking_buffer.data(),total_buffer_size);

        MPI_Bsend(buf1.data(), count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
        std::cout << "[Rank 0] buf1 sent (buffered).\n";

        MPI_Bsend(buf2.data(),count, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
        std::cout << "[Rank 0] buf2 snet (buffered).\n";

        void* temp_ptr;
        int temp_size;
        MPI_Buffer_detach(&temp_ptr, &temp_size);
    } else if (rank ==1){
        MPI_Status status;

        MPI_Recv(buf1.data(), count, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        std::cout << "[Rank 1] buf1 received. Tag:" << status.MPI_TAG << "\n";

        MPI_Recv(buf2.data(), count, MPI_FLOAT, 0 ,tag, MPI_COMM_WORLD, &status);
        std::cout << "[Rank 1] buf2 received. Tag:" << status.MPI_TAG << "\n";
        


    }
    MPI_Finalize();
    return 0;

}




