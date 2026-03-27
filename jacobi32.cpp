#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int n = 10;
    float outval = 1.23f, inval = 0.0f;
    MPI_Request req;
    MPI_Status status;

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            // 1. 發送：發射後不管 (Fire and Forget)
            MPI_Isend(&outval, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // 釋放句柄，MPI 會在背景完成發送

            // 2. 接收：必須等待，確保數據到位
            MPI_Irecv(&inval, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, &status);
            
            std::cout << "[Rank 0] Iteration " << i << " completed.\n";
        }
    } else if (rank == 1) {
        // 為了安全，先接第一球
        MPI_Irecv(&inval, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &status);

        for (int i = 0; i < n - 1; ++i) {
            // 回傳數據，釋放句柄
            MPI_Isend(&outval, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);

            // 接下一球
            MPI_Irecv(&inval, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, &status);
        }
        
        // 最後一球發送
        MPI_Isend(&outval, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &status); // 最後一球通常建議 Wait，確保程序結束前發完
    }

    MPI_Finalize();
    return 0;
}