#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows = 16;
    const int mysize = 4; // 每個進程處理的列數
    const int steps = 10;
    
    // a[列][行]：為了讓一列數據在內存中連續，方便 MPI 發送
    std::vector<std::vector<float>> a(mysize + 2, std::vector<float>(rows, 0.0f));
    std::vector<std::vector<float>> b(mysize + 2, std::vector<float>(rows, 0.0f));

    // 1. 初始化邊界條件 (與原代碼邏輯一致)
    for (int i = 0; i < rows; ++i) {
        a[0][i] = 8.0f;           // 左邊界
        a[mysize + 1][i] = 8.0f; // 右邊界
    }
    // ... 這裡省略部分初始化邏輯 ...

    // 2. 確定鄰居
    int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < 3) ? rank + 1 : MPI_PROC_NULL;

    int begin_col = (rank == 0) ? 2 : 1;
    int end_col = (rank == 3) ? mysize - 1 : mysize;

    // 3. 迭代計算
    for (int n = 0; n < steps; ++n) {
        // A. 計算邊界列 (依賴外部數據的部分)
        for (int i = 1; i < rows - 1; ++i) {
            // 計算最左邊界列
            b[1][i] = (a[0][i] + a[2][i] + a[1][i-1] + a[1][i+1]) * 0.25f;
            // 計算最右邊界列
            b[mysize][i] = (a[mysize-1][i] + a[mysize+1][i] + a[mysize][i-1] + a[mysize][i+1]) * 0.25f;
        }

        // B. 非阻塞通訊：交換邊界
        MPI_Request reqs[4];
        // 發送給右邊，接收來自左邊
        MPI_Isend(&b[mysize][0], rows, MPI_FLOAT, right, 3, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&a[0][0], rows, MPI_FLOAT, left, 3, MPI_COMM_WORLD, &reqs[1]);
        // 發送給左邊，接收來自右邊
        MPI_Isend(&b[1][0], rows, MPI_FLOAT, left, 4, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&a[mysize+1][0], rows, MPI_FLOAT, right, 4, MPI_COMM_WORLD, &reqs[3]);

        // C. 計算中間不需要通訊的部分 (重疊 Overlap)
        for (int j = 2; j < mysize; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                b[j][i] = (a[j-1][i] + a[j+1][i] + a[j][i-1] + a[j][i+1]) * 0.25f;
            }
        }

        // D. 更新數組
        for (int j = 1; j <= mysize; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                a[j][i] = b[j][i];
            }
        }

        // E. 完成所有通訊，確保下一輪開始前所有數據到位
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Finalize();
    return 0;
}