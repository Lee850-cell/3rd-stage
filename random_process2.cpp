#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdio>

int main(int argc, char** argv) {
    // -------------------------------------------------------
    // 0. 初始化與總計時開始
    // -------------------------------------------------------
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start_time = MPI_Wtime();

    // 讀取命令行參數 N，預設為 2000
    int N = 2000;
    if (argc > 1) N = std::atoi(argv[1]);
    const int max_iter = 1000;    // 測試建議跑 1000 次以獲得穩定性能
    const float BC_TEMP = 100.0f; // 邊界溫度

    // -------------------------------------------------------
    // 1. 初始化階段 (拓撲、內存、數據類型)
    // -------------------------------------------------------
    double init_start = MPI_Wtime();

    // 創建 2D 笛卡爾拓撲
    int dims[2] = {0, 0}; 
    MPI_Dims_create(world_size, 2, dims); 
    int periods[2] = {0, 0}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // 獲取上下左右鄰居的 Rank
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);    
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); 

    // 計算本地子塊大小
    int local_N = N / dims[0];
    int local_M = N / dims[1];

    // 分配 A 和 B (包含影子格)
    std::vector<float> A((local_N + 2) * (local_M + 2), 0.0f);
    std::vector<float> B((local_N + 2) * (local_M + 2), 0.0f);

    auto idx = [&](int r, int c) { return r * (local_M + 2) + c; };

    // 初始化物理邊界 (100度)
    if (up == MPI_PROC_NULL)    for (int j = 1; j <= local_M; ++j) A[idx(1, j)] = BC_TEMP;
    if (down == MPI_PROC_NULL)  for (int j = 1; j <= local_M; ++j) A[idx(local_N, j)] = BC_TEMP;
    if (left == MPI_PROC_NULL)  for (int i = 1; i <= local_N; ++i) A[idx(i, 1)] = BC_TEMP;
    if (right == MPI_PROC_NULL) for (int i = 1; i <= local_N; ++i) A[idx(i, local_M)] = BC_TEMP;

    // 定義列通訊類型 (Vector Type)
    MPI_Datatype column_type;
    MPI_Type_vector(local_N, 1, local_M + 2, MPI_FLOAT, &column_type);
    MPI_Type_commit(&column_type);

    double init_time = MPI_Wtime() - init_start;

    // -------------------------------------------------------
    // 2. 迭代計算與非阻塞通信 (核心循環)
    // -------------------------------------------------------
    double comm_time = 0.0;
    double comp_time = 0.0;
    MPI_Request reqs[8]; 

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < max_iter; ++iter) {
        
        double t_comm_start = MPI_Wtime();
        
        // A. 發起非阻塞接收 (Irecv)
        MPI_Irecv(&A[idx(0, 1)],         local_M, MPI_FLOAT, up,    10, cart_comm, &reqs[0]);
        MPI_Irecv(&A[idx(local_N+1, 1)], local_M, MPI_FLOAT, down,  11, cart_comm, &reqs[1]);
        MPI_Irecv(&A[idx(1, 0)],         1, column_type,     left,  12, cart_comm, &reqs[2]);
        MPI_Irecv(&A[idx(1, local_M+1)], 1, column_type,     right, 13, cart_comm, &reqs[3]);

        // B. 發起非阻塞發送 (Isend)
        MPI_Isend(&A[idx(1, 1)],         local_M, MPI_FLOAT, up,    11, cart_comm, &reqs[4]);
        MPI_Isend(&A[idx(local_N, 1)],   local_M, MPI_FLOAT, down,  10, cart_comm, &reqs[5]);
        MPI_Isend(&A[idx(1, 1)],         1, column_type,     left,  13, cart_comm, &reqs[6]);
        MPI_Isend(&A[idx(1, local_M)],   1, column_type,     right, 12, cart_comm, &reqs[7]);

        comm_time += (MPI_Wtime() - t_comm_start);

        // C. 【重疊優化】計算子矩陣內部 (不依賴影子格的部分)
        double t_comp_start = MPI_Wtime();
        for (int i = 2; i < local_N; ++i) {
            for (int j = 2; j < local_M; ++j) {
                B[idx(i, j)] = 0.25f * (A[idx(i - 1, j)] + A[idx(i + 1, j)] + 
                                        A[idx(i, j - 1)] + A[idx(i, j + 1)]);
            }
        }
        comp_time += (MPI_Wtime() - t_comp_start);

        // D. 等待數據到齊
        double t_wait_start = MPI_Wtime();
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
        comm_time += (MPI_Wtime() - t_wait_start);

        // E. 計算邊緣區域 (必須等通訊完才能算)
        t_comp_start = MPI_Wtime();
        for (int i = 1; i <= local_N; ++i) {
            for (int j = 1; j <= local_M; ++j) {
                // 如果是在最外層邊緣上
                if (i == 1 || i == local_N || j == 1 || j == local_M) {
                    bool is_phys_bc = ( (up == MPI_PROC_NULL && i == 1) || 
                                         (down == MPI_PROC_NULL && i == local_N) ||
                                         (left == MPI_PROC_NULL && j == 1) || 
                                         (right == MPI_PROC_NULL && j == local_M) );
                    if (is_phys_bc) {
                        B[idx(i, j)] = A[idx(i, j)]; // 保持物理邊界不變
                    } else {
                        B[idx(i, j)] = 0.25f * (A[idx(i - 1, j)] + A[idx(i + 1, j)] + 
                                                A[idx(i, j - 1)] + A[idx(i, j + 1)]);
                    }
                }
            }
        }
        comp_time += (MPI_Wtime() - t_comp_start);

        // F. 高效指針交換
        A.swap(B); 
    }

    // -------------------------------------------------------
    // 3. 數據收集與文件 IO
    // -------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double io_start = MPI_Wtime();

    std::vector<float> local_data;
    local_data.reserve(local_N * local_M);
    for (int i = 1; i <= local_N; ++i) {
        for (int j = 1; j <= local_M; ++j) local_data.push_back(A[idx(i, j)]);
    }

    if (world_rank == 0) {
        std::vector<float> full_matrix(N * N, 0.0f);
        // 放入 Rank 0 自己的數據
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < local_M; ++j) full_matrix[i * N + j] = local_data[i * local_M + j];
        }
        // 接收其他 Rank
        for (int p = 1; p < world_size; ++p) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            std::vector<float> recv_buf(local_N * local_M);
            MPI_Recv(recv_buf.data(), local_N * local_M, MPI_FLOAT, p, 99, cart_comm, MPI_STATUS_IGNORE);
            int start_r = p_coords[0] * local_N, start_c = p_coords[1] * local_M;
            for (int i = 0; i < local_N; ++i) {
                for (int j = 0; j < local_M; ++j) full_matrix[(start_r + i) * N + (start_c + j)] = recv_buf[i * local_M + j];
            }
        }
        // 寫入 TXT
        FILE* fp = fopen("result.txt", "w");
        if (fp) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) fprintf(fp, "%6.2f ", full_matrix[i * N + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    } else {
        MPI_Send(local_data.data(), local_N * local_M, MPI_FLOAT, 0, 99, cart_comm);
    }

    double io_time = MPI_Wtime() - io_start;

    // -------------------------------------------------------
    // 4. 性能數據報告
    // -------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - total_start_time;

    double max_init, max_comp, max_comm, max_io, max_total;
    MPI_Reduce(&init_time, &max_init, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&io_time, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "\n========== 性能分析報告 (非阻塞優化版) ==========\n";
        std::cout << "矩陣大小: " << N << "x" << N << " | 迭代次數: " << max_iter << "\n";
        std::cout << "進程總數: " << world_size << " (" << dims[0] << "x" << dims[1] << " 網格)\n";
        std::cout << "----------------------------------\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "初始化耗時: " << max_init << " 秒\n";
        std::cout << "純計算耗時: " << max_comp << " 秒\n";
        std::cout << "純通信耗時: " << max_comm << " 秒\n";
        std::cout << "文件IO耗時: " << max_io << " 秒\n";
        std::cout << "程序總耗時: " << max_total << " 秒\n";
        std::cout << "----------------------------------\n";
        std::cout << "並行效率分析: 通信佔比 " << (max_comm / max_total) * 100.0 << " %\n";
        std::cout << "================================================\n";
    }

    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}