#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int totalsize = 16;
    const int mysize = totalsize / 4; // 每個進程處理 4 列
    const int steps = 10;
    
    // 定義數組：a[列][行] 為了保證每一列數據在內存中是連續的
    float a[mysize + 2][totalsize];
    float b[mysize + 2][totalsize];

    // 1. 初始化數組
    for (int j = 0; j < mysize + 2; ++j) {
        for (int i = 0; i < totalsize; ++i) {
            a[j][i] = 0.0f;
            // 頂部和底部邊界賦初值 8.0
            if (i == 0 || i == totalsize - 1) a[j][i] = 8.0f;
        }
    }

    // 左右兩端進程的物理邊界賦初值
    if (myid == 0) for (int i = 0; i < totalsize; ++i) a[1][i] = 8.0f;
    if (myid == 3) for (int i = 0; i < totalsize; ++i) a[mysize][i] = 8.0f;

    // 2. 確定鄰居與迭代區間
    int left = (myid > 0) ? myid - 1 : MPI_PROC_NULL;
    int right = (myid < 3) ? myid + 1 : MPI_PROC_NULL;

    int begin_col = 1;
    int end_col = mysize;
    if (myid == 0) begin_col = 2;
    if (myid == 3) end_col = mysize - 1;

    // 3. --- 初始化持久通信 (Persistent Requests) ---
    MPI_Request req[4];
    // 標籤定義：tag1=3, tag2=4
    MPI_Send_init(&b[end_col][0], totalsize, MPI_FLOAT, right, 3, MPI_COMM_WORLD, &req[0]);
    MPI_Send_init(&b[begin_col][0], totalsize, MPI_FLOAT, left, 4, MPI_COMM_WORLD, &req[1]);
    MPI_Recv_init(&a[0][0], totalsize, MPI_FLOAT, left, 3, MPI_COMM_WORLD, &req[2]);
    MPI_Recv_init(&a[mysize + 1][0], totalsize, MPI_FLOAT, right, 4, MPI_COMM_WORLD, &req[3]);

    // 4. --- 執行迭代 ---
    for (int n = 0; n < steps; ++n) {
        // A. 計算邊界部分 (為了儘早啟動通信)
        for (int i = 1; i < totalsize - 1; ++i) {
            auto calc_point = [&](int j, int row) {
                return (a[j+1][row] + a[j-1][row] + a[row+1][j] + a[row-1][j]) * 0.25f; 
                // 注意：這裡簡化了公式演示，邏輯應與原本五點算子一致
            };
            // 實際計算邏輯：
            b[begin_col][i] = (a[begin_col+1][i] + a[begin_col-1][i] + a[begin_col][i+1] + a[begin_col][i-1]) * 0.25f;
            b[end_col][i]   = (a[end_col+1][i] + a[end_col-1][i] + a[end_col][i+1] + a[end_col][i-1]) * 0.25f;
        }

        // B. 激活所有持久通信
        MPI_Startall(4, req);

        // C. 計算中間不依賴鄰居的部分 (Overlap 計算與通信)
        for (int j = begin_col + 1; j < end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) {
                b[j][i] = (a[j+1][i] + a[j-1][i] + a[j][i+1] + a[j][i-1]) * 0.25f;
            }
        }

        // D. 更新數組
        for (int j = begin_col; j <= end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) a[j][i] = b[j][i];
        }

        // E. 等待本輪通信完成
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    }

    // 5. 打印結果
    for (int i = 1; i < totalsize - 1; ++i) {
        std::cout << "[Rank " << myid << " Row " << i << "]: ";
        for (int j = begin_col; j <= end_col; ++j) std::cout << std::fixed << std::setprecision(2) << a[j][i] << " ";
        std::cout << std::endl;
    }

    // 6. 釋放持久通信句柄
    for (int i = 0; i < 4; ++i) MPI_Request_free(&req[i]);

    MPI_Finalize();
    return 0;
}