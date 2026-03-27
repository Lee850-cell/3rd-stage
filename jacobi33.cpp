#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 3) {
        if (rank == 0) std::cerr << "This demo requires at least 3 processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        int i = 100;
        MPI_Send(&i, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 1) {
        float x = 3.14f;
        MPI_Send(&x, 1, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 2) {
        for (int i = 0; i < 2; ++i) {
            MPI_Status status;
            // 1. 先偵察消息 (不分來源，標籤為 0)
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            // 2. 根據偵察到的來源決定如何接收
            if (status.MPI_SOURCE == 0) {
                int recv_i;
                MPI_Recv(&recv_i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "[Rank 2] Received INT " << recv_i << " from Rank 0\n";
            } 
            else if (status.MPI_SOURCE == 1) {
                float recv_x;
                MPI_Recv(&recv_x, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "[Rank 2] Received FLOAT " << recv_x << " from Rank 1\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}