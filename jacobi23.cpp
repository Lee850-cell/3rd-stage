#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    const int TOTAL_SIZE = 16;
    const int STEPS = 10;
    int myid, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs != 4) {
        if (myid == 0) std::cerr << "This program requires exactly 4 processes!" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int MY_SIZE = TOTAL_SIZE / numprocs;
    // 定義局部數組 (16x6)，包含兩側護城河
    std::vector<std::vector<double>> a(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));
    std::vector<std::vector<double>> b(TOTAL_SIZE, std::vector<double>(MY_SIZE + 2, 0.0));

    // --- 1. 數組初始化 ---
    if (myid == 0) {
        for (int i = 0; i < TOTAL_SIZE; i++) a[i][1] = 8.0; // 左牆
    }
    if (myid == 3) {
        for (int i = 0; i < TOTAL_SIZE; i++) a[i][MY_SIZE] = 8.0; // 右牆
    }
    for (int j = 0; j < MY_SIZE + 2; j++) {
        a[0][j] = 8.0;              // 頂牆
        a[TOTAL_SIZE - 1][j] = 8.0;  // 底牆
    }

    // --- 2. 設定鄰居標識 (核心改進) ---
    int tag1 = 3, tag2 = 4;
    int left = (myid > 0) ? myid - 1 : MPI_PROC_NULL;
    int right = (myid < 3) ? myid + 1 : MPI_PROC_NULL;

    // --- 3. Jacobi 迭代 ---
    for (int n = 0; n < STEPS; n++) {
        MPI_Status status;
        std::vector<double> send_buf(TOTAL_SIZE);
        std::vector<double> recv_buf(TOTAL_SIZE);

        // A. 從左向右平移數據 (發送右邊界給 right，從 left 接收放到左護城河)
        for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][MY_SIZE];
        MPI_Sendrecv(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, right, tag1,
                     recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, left, tag1,
                     MPI_COMM_WORLD, &status);
        
        // 如果不是 MPI_PROC_NULL，就更新左護城河
        if (left != MPI_PROC_NULL) {
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][0] = recv_buf[i];
        }

        // B. 從右向左平移數據 (發送左邊界給 left，從 right 接收放到右護城河)
        for (int i = 0; i < TOTAL_SIZE; i++) send_buf[i] = a[i][1];
        MPI_Sendrecv(send_buf.data(), TOTAL_SIZE, MPI_DOUBLE, left, tag2,
                     recv_buf.data(), TOTAL_SIZE, MPI_DOUBLE, right, tag2,
                     MPI_COMM_WORLD, &status);
        
        // 如果不是 MPI_PROC_NULL，就更新右護城河
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < TOTAL_SIZE; i++) a[i][MY_SIZE + 1] = recv_buf[i];
        }

        // --- 4. 計算與更新 ---
        int begin_col = 1, end_col = MY_SIZE;
        if (myid == 0) begin_col = 2;
        if (myid == 3) end_col = MY_SIZE - 1;

        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) {
                b[i][j] = (a[i][j+1] + a[i][j-1] + a[i+1][j] + a[i-1][j]) * 0.25;
            }
        }
        for (int j = begin_col; j <= end_col; j++) {
            for (int i = 1; i < TOTAL_SIZE - 1; i++) a[i][j] = b[i][j];
        }
    }

    // --- 5. 輸出結果 ---
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 1; i < TOTAL_SIZE - 1; i++) {
        std::cout << "Rank " << myid << " Row " << i << ": ";
        int bc = (myid == 0) ? 2 : 1;
        int ec = (myid == 3) ? MY_SIZE - 1 : MY_SIZE;
        for (int j = bc; j <= ec; j++) std::cout << a[i][j] << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}