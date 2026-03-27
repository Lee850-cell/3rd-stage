#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    // 1. 初始化環境
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // --- 全局參數 ---
    const int N = 2000;          // 全局矩陣大小 N*N
    const int max_iter = 1000;    // 迭代步數
    const float BC_TEMP = 100.0f; // 邊界溫度

    // 2. 創建二維笛卡爾拓撲 (2D Domain Decomposition)
    int dims[2] = {0, 0}; 
    MPI_Dims_create(world_size, 2, dims); // 自動分配 P_rows x P_cols
    
    int periods[2] = {0, 0}; // 非週期性邊界
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // 獲取當前進程在 2D 網格中的座標
    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // 獲取上下左右鄰居的 Rank (如果不存在則返回 MPI_PROC_NULL)
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);    // 維度 0: 行方向 (上下)
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // 維度 1: 列方向 (左右)

    // 3. 計算本地子矩陣大小 (假設 N 可被整除，否則需額外處理餘數)
    int local_N = N / dims[0];
    int local_M = N / dims[1];

    // 包含影子格的本地緩衝區 (local_N+2) x (local_M+2)
    std::vector<float> A((local_N + 2) * (local_M + 2), 0.0f);
    std::vector<float> B((local_N + 2) * (local_M + 2), 0.0f);

    // 4. 輔助 Lambda 函數：二維索引轉一維
    auto idx = [&](int r, int c) {
        return r * (local_M + 2) + c;
    };

    // 5. 初始化物理邊界 (BC)
    // 只有位於矩陣最外層邊緣的進程才設置 100 度
    if (up == MPI_PROC_NULL) {
        for (int j = 1; j <= local_M; ++j) A[idx(1, j)] = BC_TEMP;
    }
    if (down == MPI_PROC_NULL) {
        for (int j = 1; j <= local_M; ++j) A[idx(local_N, j)] = BC_TEMP;
    }
    if (left == MPI_PROC_NULL) {
        for (int i = 1; i <= local_N; ++i) A[idx(i, 1)] = BC_TEMP;
    }
    if (right == MPI_PROC_NULL) {
        for (int i = 1; i <= local_N; ++i) A[idx(i, local_M)] = BC_TEMP;
    }

    // 用於列交換的臨時緩衝區 (C++ 內存是行連續的，列不連續，需要手動打包)
    std::vector<float> col_send(local_N);
    std::vector<float> col_recv(local_N);

    // 6. 迭代主循環
    for (int iter = 0; iter < max_iter; ++iter) {
        
        // --- A. 數據交換 (影子格交換) ---
        
        // 1. 上下交換 (行交換，內存連續)
        // 向下發送最後一行真實數據，接收上方影子格
        MPI_Sendrecv(&A[idx(local_N, 1)], local_M, MPI_FLOAT, down, 0,
                     &A[idx(0, 1)],       local_M, MPI_FLOAT, up,   0, 
                     cart_comm, MPI_STATUS_IGNORE);
        // 向上發送第一行真實數據，接收下方影子格
        MPI_Sendrecv(&A[idx(1, 1)],       local_M, MPI_FLOAT, up,   1,
                     &A[idx(local_N+1, 1)], local_M, MPI_FLOAT, down, 1, 
                     cart_comm, MPI_STATUS_IGNORE);

        // 2. 左右交換 (列交換，內存不連續，需手動處理)
        // 向右發送最後一列，接收左方影子格
        for(int i=1; i<=local_N; ++i) col_send[i-1] = A[idx(i, local_M)];
        MPI_Sendrecv(col_send.data(), local_N, MPI_FLOAT, right, 2,
                     col_recv.data(), local_N, MPI_FLOAT, left,  2, 
                     cart_comm, MPI_STATUS_IGNORE);
        if (left != MPI_PROC_NULL) {
            for(int i=1; i<=local_N; ++i) A[idx(i, 0)] = col_recv[i-1];
        }

        // 向左發送第一列，接收右方影子格
        for(int i=1; i<=local_N; ++i) col_send[i-1] = A[idx(i, 1)];
        MPI_Sendrecv(col_send.data(), local_N, MPI_FLOAT, left,  3,
                     col_recv.data(), local_N, MPI_FLOAT, right, 3, 
                     cart_comm, MPI_STATUS_IGNORE);
        if (right != MPI_PROC_NULL) {
            for(int i=1; i<=local_N; ++i) A[idx(i, local_M+1)] = col_recv[i-1];
        }

        // --- B. Jacobi 核心計算 ---
        // 只計算內部真實數據區，邊界保持不變（如果是物理邊界的話）
        for (int i = 1; i <= local_N; ++i) {
            for (int j = 1; j <= local_M; ++j) {
                // 如果是物理邊界點，跳過更新（保持 100 度）
                bool is_boundary = ( (up == MPI_PROC_NULL && i == 1) || 
                                     (down == MPI_PROC_NULL && i == local_N) ||
                                     (left == MPI_PROC_NULL && j == 1) || 
                                     (right == MPI_PROC_NULL && j == local_M) );
                
                if (is_boundary) {
                    B[idx(i, j)] = A[idx(i, j)];
                } else {
                    B[idx(i, j)] = 0.25f * (A[idx(i - 1, j)] + A[idx(i + 1, j)] + 
                                            A[idx(i, j - 1)] + A[idx(i, j + 1)]);
                }
            }
        }
        
        // 緩衝區交換 (指針交換在 vector 中可用 A = B 代替，但注意 A = B 會發生拷貝)
        A = B;

        // 每 10 步打印一次進度
        if (world_rank == 0 && iter % 10 == 0) {
            std::cout << "Iteration " << iter << " completed." << std::endl;
        }
    }

    // 7. 結束
    if (world_rank == 0) {
        std::cout << "Jacobi solver finished successfully for " << N << "x" << N << " grid." << std::endl;
        std::cout << "Process Layout: " << dims[0] << "x" << dims[1] << std::endl;
    }
    // ... 前面是 MPI 初始化、子矩陣分配和 100 步迭代循環 ...

    for (int iter = 0; iter < max_iter; ++iter) {
        // [這裡是你原本的邊界交換和 Jacobi 計算代碼]
    }

    // ===========================================================
    // 這裡開始插入「輸出到文件」的代碼
    // ===========================================================

    // 1. 準備本地數據（去掉影子格，只取中間真實的 local_N * local_M 部分）
    std::vector<float> local_data;
    local_data.reserve(local_N * local_M);
    for (int i = 1; i <= local_N; ++i) {
        for (int j = 1; j <= local_M; ++j) {
            local_data.push_back(A[idx(i, j)]);
        }
    }

    if (world_rank == 0) {
        // --- Rank 0 的工作：創建大地圖並收集中碎片 ---
        std::vector<float> full_matrix(N * N, 0.0f);

        // A. 先把 Rank 0 自己的數據填進去
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < local_M; ++j) {
                full_matrix[i * N + j] = local_data[i * local_M + j];
            }
        }

        // B. 接收來自其他所有進程 (Rank 1 到 world_size-1) 的數據
        for (int p = 1; p < world_size; ++p) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords); // 查出那個進程在哪個座標
            
            std::vector<float> recv_buf(local_N * local_M);
            MPI_Recv(recv_buf.data(), local_N * local_M, MPI_FLOAT, p, 99, cart_comm, MPI_STATUS_IGNORE);

            // 根據座標算出它在全局大矩陣中的起始位置
            int start_r = p_coords[0] * local_N;
            int start_c = p_coords[1] * local_M;
            for (int i = 0; i < local_N; ++i) {
                for (int j = 0; j < local_M; ++j) {
                    full_matrix[(start_r + i) * N + (start_c + j)] = recv_buf[i * local_M + j];
                }
            }
        }

        // C. 統一寫入 TXT 文件
        std::cout << "Writing to result.txt..." << std::endl;
        FILE* fp = fopen("result.txt", "w");
        if (fp) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    fprintf(fp, "%6.2f ", full_matrix[i * N + j]);
                }
                fprintf(fp, "\n"); // 每一行寫完換行，保持 2000*2000 形狀
            }
            fclose(fp);
            std::cout << "Done! File saved." << std::endl;
        }

    } else {
        // --- 其他 Rank 的工作：把自己的數據發給 Rank 0 ---
        MPI_Send(local_data.data(), local_N * local_M, MPI_FLOAT, 0, 99, cart_comm);
    }

    // ===========================================================
    // 輸出結束
    // ===========================================================

    MPI_Finalize();
    return 0;
}