#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int val = 100;
        MPI_Send(&val, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 1) {
        float val = 3.14f;
        MPI_Send(&val, 1, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 2) {
        for (int i = 0; i < 2; ++i) {
            MPI_Status status;
            
            // 1. 偵察：是誰發的消息？
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            // 2. 鎖定來源：從 status 中提取確切的發送者
            int sender = status.MPI_SOURCE;

            if (sender == 0) {
                int recv_val;
                // 正確做法：指定接收來自 sender (Rank 0) 的消息
                MPI_Recv(&recv_val, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "[Rank 2] Safely received INT from Rank 0: " << recv_val << std::endl;
            } 
            else if (sender == 1) {
                float recv_val;
                // 正確做法：指定接收來自 sender (Rank 1) 的消息
                MPI_Recv(&recv_val, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "[Rank 2] Safely received FLOAT from Rank 1: " << recv_val << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}