#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "This program requires exactly 2 processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int MAX_CAPACITY = 10;
    std::vector<int> buffer(MAX_CAPACITY, 0);

    if (rank == 0) { // 發送端
        // 第一次同步發送：1 個整數 (Tag 1)
        int send_count = 1;
        std::cout << "[Rank 0] Ssending " << send_count << " data (Tag 1)...\n";
        MPI_Ssend(buffer.data(), send_count, MPI_INT, 1, 1, MPI_COMM_WORLD);

        // 第二次同步發送：4 個整數 (Tag 2)
        send_count = 4;
        std::cout << "[Rank 0] Ssending " << send_count << " data (Tag 2)...\n";
        MPI_Ssend(buffer.data(), send_count, MPI_INT, 1, 2, MPI_COMM_WORLD);

    } else { // 接收端 (Rank 1)
        for (int i = 1; i <= 2; ++i) {
            MPI_Status status;
            // 每次嘗試接收最多 5 個元素
            MPI_Recv(buffer.data(), 5, MPI_INT, 0, i, MPI_COMM_WORLD, &status);

            int actual_count;
            MPI_Get_count(&status, MPI_INT, &actual_count);
            
            std::cout << "[Rank 1] Received message with Tag " << status.MPI_TAG 
                      << ", Actual count: " << actual_count << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}