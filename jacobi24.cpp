#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    int myid, numprocs;
    const int ROWS = 100;
    const int COLS = 100;
    const int MASTER = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    std::vector<double> b(COLS);
    
    if (myid == MASTER) {
        // --- 主進程初始化 ---
        std::vector<std::vector<double>> a(ROWS, std::vector<double>(COLS));
        std::vector<double> c(ROWS);
        for (int i = 0; i < ROWS; i++) {
            b[i] = 1.0;
            for (int j = 0; j < COLS; j++) a[i][j] = i + 1; // 對應 Fortran 的 a(i,j)=i
        }

        // 1. 廣播向量 B
        MPI_Bcast(b.data(), COLS, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        int numsent = 0;
        int numrcvd = 0;

        // 2. 初始分發：給每個從進程發一行
        for (int i = 1; i < numprocs && i <= ROWS; i++) {
            MPI_Send(a[numsent].data(), COLS, MPI_DOUBLE, i, numsent + 1, MPI_COMM_WORLD);
            numsent++;
        }

        // 3. 動態循環：收一個，發一個
        for (int i = 0; i < ROWS; i++) {
            double ans;
            MPI_Status status;
            MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            int sender = status.MPI_SOURCE;
            int row_index = status.MPI_TAG; // 這裡 Tag 代表行號
            c[row_index - 1] = ans;

            if (numsent < ROWS) {
                // 還有沒算的，繼續派活
                MPI_Send(a[numsent].data(), COLS, MPI_DOUBLE, sender, numsent + 1, MPI_COMM_WORLD);
                numsent++;
            } else {
                // 沒活了，發送終止信號 (Tag=0)
                MPI_Send(NULL, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
            }
        }

        // 打印部分結果驗證
        std::cout << "Calculation finished. C[0] = " << c[0] << ", C[99] = " << c[99] << std::endl;

    } else {
        // --- 從進程 (Worker) ---
        MPI_Bcast(b.data(), COLS, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        std::vector<double> buffer(COLS);
        while (true) {
            MPI_Status status;
            MPI_Recv(buffer.data(), COLS, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == 0) break; // 收到毒丸，下班

            int row_index = status.MPI_TAG;
            double ans = 0.0;
            for (int i = 0; i < COLS; i++) {
                ans += buffer[i] * b[i]; // 計算內積
            }

            // 將結果發回給主進程，並帶上行號 Tag
            MPI_Send(&ans, 1, MPI_DOUBLE, MASTER, row_index, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}