#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float a = 0.3f, b = 0.4f;
    // 定義兩個請求句柄，分別對應 a 和 b 的通訊
    MPI_Request reqs[2];
    MPI_Status stats[2];

    if (rank == 0) {
        MPI_Isend(&a, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&b, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &reqs[1]);

    } else if (rank == 1) {
        MPI_Irecv(&a, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(&b, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqs[1]);
    }

    MPI_Wait(&reqs[0], &stats[0]);
    MPI_Wait(&reqs[1], &stats[1]);

    std::cout << "a = " << a << ", b = " << b << std::endl;

    MPI_Finalize();


    return 0;
}